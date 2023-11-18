"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import tqdm

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from robomimic.scripts.train_lin_dynamics import LINATTRACT


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    
    if args.add_perturbation:
        env_perturber = EnvPerturber()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            act = policy(ob=obs)
            
            if args.use_attractor:
                env.env._update_mode_cache()
                current_mode = env.env.get_current_mode() # use ground truth mode for now
                if env.name == "PickPlaceCan":
                    ## Action: [eef_xyz (3), eef_rot (3), gripper (1)]
                    eef_pos = torch.Tensor(obs["robot0_eef_pos"])
                    eef_pos.requires_grad_(True)
                    obj_pos = torch.from_numpy(obs["object"][:3])
                    dist_eef_obj = torch.linalg.norm(obj_pos.data - eef_pos.data, 2)
                    if current_mode.name == "free":
                        attractor = torch.Tensor([-0.00016607, 0.00097243, 0.01710138]) # relative
                        attractor[2] *= 1.5 # NOTE: place the attractor a bit higher
                        potential_T = 1.
                        coef = 0.1
                        min_dist_eef_obj = 0.1
                        def potential_fn(_eef_pos):
                            _x_in_attractor_space = obj_pos - _eef_pos
                            dist = torch.linalg.norm(_x_in_attractor_space - attractor, 2)
                            # dist = torch.abs(_x_in_attractor_space - attractor).mean()
                            return torch.exp(dist / potential_T)
                    elif current_mode.name == "grasping":
                        attractor = torch.Tensor([0.18300294, 0.2530928, 1.03142587])
                        potential_T = 1.
                        coef = 0.05
                        min_dist_eef_obj = 99. # NOTE: not using attractor controller
                        def potential_fn(_eef_pos):
                            _x_in_attractor_space = _eef_pos
                            dist = torch.linalg.norm(_x_in_attractor_space - attractor, 2)
                            # dist = torch.abs(_x_in_attractor_space - attractor).mean()
                            return torch.exp(dist / potential_T)
                        potential = potential_fn(eef_pos)
                        grad = torch.autograd.grad(potential, eef_pos)[0]
                        act[:3] += -coef * grad.numpy()
                    elif current_mode.name == "hovering":
                        attractor = torch.Tensor([0.20470058, 0.31206185, 1.07567048])
                        potential_T = 1.
                        coef = 0.05
                        min_dist_eef_obj = 99. # NOTE: not using attractor controller
                        def potential_fn(_eef_pos):
                            _x_in_attractor_space = _eef_pos
                            dist = torch.linalg.norm(_x_in_attractor_space - attractor, 2)
                            # dist = torch.abs(_x_in_attractor_space - attractor).mean()
                            return torch.exp(dist / potential_T)
                    
                    potential = potential_fn(eef_pos)
                    grad = torch.autograd.grad(potential, eef_pos)[0]
                    if dist_eef_obj >= min_dist_eef_obj:
                        act[:3] += coef * -grad.numpy()
                        if args.use_upright_gripper:
                            act[3:6] = torch.Tensor([0, 0, 0])
                        # act[:3] = coef * -grad.numpy() # NOTE: overwrite
                        # print("overwrite / modify action") # DEBUG
                elif env.name == "Lift":
                    eef_pos = torch.Tensor(obs["robot0_eef_pos"])
                    eef_pos.requires_grad_(True)
                    obj_pos = torch.from_numpy(obs["object"][:3])
                    dist_eef_obj = torch.linalg.norm(obj_pos.data - eef_pos.data, 2)
                    if current_mode.name == "free":
                        attractor = torch.Tensor([-0.00099187, -0.00120366, -0.00319776]) # relative
                        attractor[2] *= 1.5 # NOTE: place the attractor a bit higher
                        potential_T = 1.
                        coef = 0.1
                        min_dist_eef_obj = 0.1
                        def potential_fn(_eef_pos):
                            _x_in_attractor_space = obj_pos - _eef_pos
                            dist = torch.linalg.norm(_x_in_attractor_space - attractor, 2)
                            # dist = torch.abs(_x_in_attractor_space - attractor).mean()
                            return torch.exp(dist / potential_T)
                    elif current_mode.name == "grasping":
                        attractor = torch.Tensor([-0.00689444, 0.00194621, 0.87560444])
                        potential_T = 1.
                        coef = 0.05
                        min_dist_eef_obj = 99. # NOTE: not using attractor controller
                        def potential_fn(_eef_pos):
                            _x_in_attractor_space = _eef_pos
                            dist = torch.linalg.norm(_x_in_attractor_space - attractor, 2)
                            # dist = torch.abs(_x_in_attractor_space - attractor).mean()
                            return torch.exp(dist / potential_T)
                        potential = potential_fn(eef_pos)
                        grad = torch.autograd.grad(potential, eef_pos)[0]
                        act[:3] += -coef * grad.numpy()
                    else:
                        coef = 0.05
                        min_dist_eef_obj = 99. # NOTE: not using attractor controller
                        def potential_fn(_eef_pos):
                            return _eef_pos.mean() * 0.
                    
                    potential = potential_fn(eef_pos)
                    grad = torch.autograd.grad(potential, eef_pos)[0]
                    if dist_eef_obj >= min_dist_eef_obj:
                        act[:3] += coef * -grad.numpy()
                        if args.use_upright_gripper:
                            act[3:6] = torch.Tensor([0, 0, 0])
                elif env.name == "NutAssemblySquare":
                    eef_pos = torch.Tensor(obs["robot0_eef_pos"])
                    eef_pos.requires_grad_(True)
                    obj_pos = torch.from_numpy(obs["object"][:3])
                    dist_eef_obj = torch.linalg.norm(obj_pos.data - eef_pos.data, 2)
                    if current_mode.name == "free":
                        # handle_site_pos = env.env.sim.data.get_site_xpos("SquareNut_handle_site")
                        # gripper_pos = env.env.sim.data.get_site_xpos(env.env.robots[0].gripper.important_sites["grip_site"])
                        # attractor = torch.Tensor(handle_site_pos - gripper_pos)
                        attractor = torch.Tensor([-0.00109405, -0.00054479, 0.01984971]) # relative
                        attractor[2] *= 1.5 # NOTE: place the attractor a bit higher
                        potential_T = 1.
                        coef = 0.1
                        min_dist_eef_obj = 0.1
                        def potential_fn(_eef_pos):
                            _x_in_attractor_space = obj_pos - _eef_pos
                            dist = torch.linalg.norm(_x_in_attractor_space - attractor, 2)
                            # dist = torch.abs(_x_in_attractor_space - attractor).mean()
                            return torch.exp(dist / potential_T)
                    elif current_mode.name == "grasping":
                        # attractor = torch.Tensor([0.16372011, 0.11015255, 0.96184433])
                        attractor = torch.Tensor([0.16372011, 0.11015255, 0.90184433])
                        attractor[2] *= 1. # NOTE: place the attractor a bit higher
                        potential_T = 1.
                        coef = 0.05
                        min_dist_eef_obj = 0.1 # NOTE: not using attractor controller
                        def potential_fn(_eef_pos):
                            _x_in_attractor_space = _eef_pos
                            dist = torch.linalg.norm(_x_in_attractor_space - attractor, 2)
                            # dist = torch.abs(_x_in_attractor_space - attractor).mean()
                            return torch.exp(dist / potential_T)
                        potential = potential_fn(eef_pos)
                        grad = torch.autograd.grad(potential, eef_pos)[0]
                        act[:3] += -coef * grad.numpy()
                    else:
                        coef = 0.05
                        min_dist_eef_obj = 99. # NOTE: not using attractor controller
                        def potential_fn(_eef_pos):
                            return _eef_pos.mean() * 0.
                    
                    potential = potential_fn(eef_pos)
                    grad = torch.autograd.grad(potential, eef_pos)[0]
                    if dist_eef_obj >= min_dist_eef_obj:
                        act[:3] += coef * -grad.numpy()
                        if args.use_upright_gripper:
                            act[3:6] = torch.Tensor([0, 0, 0])
                else:
                    raise ValueError(f"No attractor implemented for env {env.name}")

            # play action
            if args.add_perturbation:
                next_obs, r, done, _, is_perturbed = env_perturber(env, act)
            else:
                next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    if args.lin_policy:
        rand_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
        policy = torch.load(args.lin_path)
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=args.env, 
            render=args.render, 
            render_offscreen=(args.video_path is not None), 
            verbose=True,
        )
        # env = EnvUtils.create_env_from_metadata(env_meta=policy.policy.env_meta, render=args.render, render_offscreen=write_video, use_image_obs=False)
        
        
        rollout_horizon = args.horizon
        rollout_num_episodes = args.n_rollouts
        if rollout_horizon is None:
            rollout_horizon = 400 # can't load from checkpoint
    else:
        # restore policy
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

        # read rollout settings
        rollout_num_episodes = args.n_rollouts
        rollout_horizon = args.horizon
        if rollout_horizon is None:
            # read horizon from config
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            rollout_horizon = config.experiment.rollout.horizon

        # create environment from saved checkpoint
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=args.env, 
            render=args.render, 
            render_offscreen=(args.video_path is not None), 
            verbose=True,
        )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    pbar = tqdm.tqdm(range(rollout_num_episodes), total=rollout_num_episodes)
    for i in pbar:
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)
        success_rate = np.mean(TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)["Success_Rate"])
        pbar.set_description("Running success rate {:.2f}".format(success_rate))

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


class EnvPerturber:
    DEFAULT_CONFIG = dict(
        # probability to instantiate perturbation at each step
        perturb_ee_prob=0.1,
        perturb_grasp_prob=0.0, # 0.05,
        # the duration of each perturbation sequence
        ee_perturb_len=10,
        grasp_perturb_len=10,
        # maximal number of perturbation sequences to be applied; set to -1 for no maximum
        max_perturb_ee_cnt=2,
        max_perturb_grasp_cnt=3,
    )
    
    def __init__(self, config=dict()):
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(config)
        
        self.perturb_fn = dict(
            ee=self._perturb_ee,
            grasp=self._perturb_grasp,
        )
        
        self.reset()
        
    def reset(self):
        self.history = dict(
            # used for doing a sequence of perturbation
            ee_perturb_step=0,
            grasp_perturb_step=0,
            # used for checking if exceeding a maximal number of perturbation
            ee_perturb_cnt=0,
            grasp_perturb_cnt=0,
        )
        self.cache = dict(
            ee_perturb_target=None
        )
        
    def _perturb_ee(self, env, act):
        orig_ee = act[:-1]
        perturbed_ee = orig_ee.copy()
        if self.history["ee_perturb_step"] == 0:
            self.cache["ee_perturb_target"] = np.random.uniform(low=-1., high=1., size=3)
            self.cache["ee_perturb_target"][2] = max(0.5, self.cache["ee_perturb_target"][2])
        dir_to_target = (self.cache["ee_perturb_target"] - perturbed_ee[:3])
        dir_to_target = dir_to_target / np.linalg.norm(dir_to_target, 2)
        perturbed_ee[:3] += dir_to_target * 1.
        
        act[:-1] = perturbed_ee
        
        return act
    
    def _perturb_grasp(self, env, act):
        act[-1] = -1 # -1 is open and 1 is close
        return act
        
    def __call__(self, env, act):
        # robot = np.random.choice(env.env.robots) # suppose there are multiple robots, randomly pick one
        new_act = act.copy()
        
        is_perturbed = dict()
        for name in ["ee", "grasp"]:
            if self.config[f"perturb_{name}_prob"] > 0:
                # check if perturb or not; always perturb when the previous step is perturbed
                if self.history[f"{name}_perturb_step"] > 0:
                    perturb = True
                else:
                    perturb = np.random.uniform(0., 1.) < self.config[f"perturb_{name}_prob"]
                    
                if (self.config[f"max_perturb_{name}_cnt"] > 0) and \
                    (self.history[f"{name}_perturb_cnt"] >= self.config[f"max_perturb_{name}_cnt"]):
                    perturb = False
                    
                # apply perturbation and log step
                if perturb:
                    new_act = self.perturb_fn[name](env, new_act)
                    
                    self.history[f"{name}_perturb_step"] += 1
                    
                # reset history after a sequence of perturbation
                if self.history[f"{name}_perturb_step"] >= self.config[f"{name}_perturb_len"]:
                    self.history[f"{name}_perturb_step"] = 0
                    self.history[f"{name}_perturb_cnt"] += 1 # count only when a sequence of perturbation is complete
            else:
                perturb = False
        
            is_perturbed[name] = perturb

        next_obs, rew, done, info = env.step(new_act)
    
        return next_obs, rew, done, info, is_perturbed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    # for loading linear policies (doesn't use full robomimic env)
    parser.add_argument(
        "--lin_policy",
        action="store_true",
        help="(optional) load from linear policy",
    )

   # Path to trained model
    parser.add_argument(
        "--lin_path",
        type=str,
        required=False,
        help="path to saved checkpoint pth file for linear policy",
    )
    
    parser.add_argument(
        "--add-perturbation",
        action="store_true",
        default=False,
        help="Add perturbation to the robot (as if there is external force or effect) when doing rollout",
    )
    
    parser.add_argument(
        "--use-attractor",
        action="store_true",
        default=False,
        help="Use attractor to modify action from a pretrained policy",
    )
    
    parser.add_argument(
        "--use-upright-gripper",
        action="store_true",
        default=False,
        help="Make the gripper upright when using the attractor controller",
    )

    args = parser.parse_args()
    run_trained_agent(args)

