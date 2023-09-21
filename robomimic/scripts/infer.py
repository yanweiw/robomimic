"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use_obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use_obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use_actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use_actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use_obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import random

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType
from robosuite.wrappers import VisualizationWrapper
import sys 
sys.path.append('../../../mode_learning')
import eval 
import torch
import cv2


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

mode_colors = [[1.0, 0.0, 0.0], 
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0], 
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.6, 0.2, 0.2], 
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6]]


def downsample_array(original_array, fixed_size):
    # Ensure the fixed_size is at least 2 to include the first and last elements
    assert fixed_size >= 2
    fixed_size = max(fixed_size, 2)
    # Generate the indices to sample from the original array
    indices = np.linspace(0, len(original_array) - 1, fixed_size, dtype=int)
    # Select the elements at the generated indices
    downsampled_array = original_array[indices]
    return downsampled_array


def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    orig_pos = None,
    actions=None, 
    render=False, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
    demo_idx =None, 
    sample_size = None,
    data_save_path = None,
    mode_pred_states = None, # states used for mode classification
    model = None,
    guess_idx=0,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    # downsample the trajectory to a fixed size for visualization
    assert sample_size is not None
    orig_idx = np.array(range(states.shape[0]))
    if action_playback: 
        # save the first half of sites for original data; and the second half for perturbed data
        sampled_idx = downsample_array(orig_idx, sample_size//2)
    else:
        sampled_idx = downsample_array(orig_idx, sample_size)
    
    # plot the original ee positions as a reference
    ic_list = []
    if action_playback:
        for i, ee_pos in enumerate(orig_pos):
            if i in sampled_idx:
                in_demo_idx = np.where(sampled_idx == i)[0][0]
                ic_idx = demo_idx * sample_size + in_demo_idx
                env.env.set_indicator_pos("site{}".format(ic_idx), ee_pos)
                ic_list.append("site{}".format(ic_idx))
                # env.env.sim.forward()
        env.reset_to({"states": states[0]})

    # acculate data for each step
    keys = list(env.env.observation_spec().keys())
    keys.append('states')
    dict_of_arrays = {key: [] for key in keys}

    mode_pred_states = torch.tensor(mode_pred_states, dtype=torch.float32).unsqueeze(0)
    traj_len = mode_pred_states.shape[1]
    with torch.no_grad():
        mode, mode_log, mode_logits = model.net.pred_mode(mode_pred_states.cuda())
    mode = mode.reshape(-1, traj_len, model.net.num_guess, model.net.num_modes)[:, :, guess_idx, :]
    # color_matrix = torch.tensor(mode_colors)[:model.net.num_modes, :]
    # mode = torch.matmul(mode, color_matrix.cuda())
    # mode = torch.clamp(mode, min=0, max=1)
    mode_idx = torch.argmax(mode, dim=-1)
    # assert mode_idx.shape == mode.shape[:2]
    mode_idx = mode_idx.detach().cpu().numpy()[0] # indexing to remove batch dim

    # render the simulation
    for i in range(len(states)):
        if not action_playback:
            env.reset_to({"states" : states[i]})
        else:
            env.step(actions[i])
            # if i < len(states) - 1:
            #     # check whether the actions deterministically lead to the same recorded states
            #     state_playback = env.get_state()["states"]
            #     if not np.all(np.equal(states[i + 1], state_playback)):
            #         err = np.linalg.norm(states[i + 1] - state_playback)
            #         print("warning: playback diverged by {} at step {}".format(err, i))
            if data_save_path is not None:
                # save the data for each step
                dict_of_arrays['states'].append(env.get_state()["states"])
                obs = env.env.observation_spec()
                for key in obs.keys():
                    dict_of_arrays[key].append(obs[key])

        if i in sampled_idx:
            in_demo_idx = np.where(sampled_idx == i)[0][0]
            ic_idx = in_demo_idx # demo_idx * sample_size + in_demo_idx
            if action_playback:
                ic_idx += sample_size//2
            # env.env.set_indicator_pos("site{}".format(ic_idx), env.env._get_observations(force_update=True)["robot0_eef_pos"])
            
            env.env.set_indicator_pos("mode_{}_{}".format(mode_idx[i], ic_idx), env.env._get_observations(force_update=True)["robot0_eef_pos"])
            # print("setting indiciator sites{}".format(ic_idx))
            # ic_list.append("site{}".format(ic_idx))
            ic_list.append("mode_{}_{}".format(mode_idx[i], ic_idx))
            # env.env.sim.forward()

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    orig_img = env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                    # boundary_color = mode[i] * 255
                    boundary_color = np.array(mode_colors[mode_idx[i]]) * 255
                    orig_img[:20, :] = boundary_color
                    video_img.append(orig_img)
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        if first:
            break

    # remove the indicator sites to reduce clutter
    # if action_playback:
    for ic in ic_list:
        env.env.set_indicator_pos(ic, [0, 0, 0])

    if data_save_path is not None:
        dict_of_arrays = {key: np.vstack(dict_of_arrays[key]) for key in dict_of_arrays.keys()}
        return dict_of_arrays, env.get_reward()
    else:
        return None, None

def playback_trajectory_with_obs(
    traj_grp,
    video_writer, 
    video_skip=5, 
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["actions"].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break

def perturb_traj(orig, pert_range=0.1):
    # orig actions (traj_len, 7), this is perturbation in the joint space
    assert len(orig) > 10
    impulse_start = random.randint(0, len(orig)-10)
    impulse_end = random.randint(impulse_start+8, len(orig)-1)
    impulse_mean = (impulse_start + impulse_end)//2
    impulse_mean_action = orig[impulse_mean]
    impulse_targets = []
    for curr in impulse_mean_action:
        target = random.uniform(curr-pert_range, curr+pert_range)
        # if target < -1: target = -1
        # if target > 1: target = 1
        impulse_targets.append(target)
    # impulse_target_x = random.uniform(-8, 8)
    # impulse_target_y = random.uniform(-8, 8)
    max_relative_dist = 5 # np.exp(-5) ~= 0.006

    kernel = np.exp(-max_relative_dist*(np.array(range(len(orig))) - impulse_mean)**2 / ((impulse_start-impulse_mean)**2))
    perturbed = orig.copy()
    for i in range(orig.shape[1]):
        perturbed[:, i] += (impulse_targets[i]-perturbed[:, i])*kernel

    return perturbed


def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        # from IPython import embed; embed()
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        # directly control ee pose
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        env_meta['env_kwargs']['controller_configs']['control_ori'] = False
        env_meta['env_kwargs']['controller_configs']['kp'] = 1000
        env_meta['env_kwargs']['controller_configs']['kp_limits'] = [0, 1000]
        env_meta['env_kwargs']['controller_configs']['output_max'] = [2, 2, 2, 1, 1, 1, 1] # these values are just placeholders
        env_meta['env_kwargs']['controller_configs']['output_min'] = [-2, -2, -2, -1, -1, -1, -1]        
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")
    # from IPython import embed; embed()
    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())

    if args.fail:
        demos = [demo for demo in demos if 'fail' in demo]
    elif args.succ:
        demos = [demo for demo in demos if 'succ' in demo]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[args.n:args.n+60]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    if not args.use_obs:
        sample_size = args.ic
        if args.use_actions:
            sample_size = sample_size * 2

        ic = []
        for i in range(1): #range(len(demos)):
            for mode_idx, color in enumerate(mode_colors):
                rgba = color + [0.5]
                ic += [
                    {
                    "type": "sphere",
                    "size": [0.004],
                    "rgba": rgba,
                    "name": "mode_{}_{}".format(mode_idx, i * sample_size + j),
                    }
                    for j in range(sample_size)
                ]

        # ic = []
        # for i in range(len(demos)):
        #     rgba_random = np.random.uniform(0, 1, 3).tolist() + [0.5]
        #     blue = [0, 0, 1, 1] 
        #     red = [1, 0, 0, 1]
        #     if args.use_actions:
        #         rgba1 = blue
        #         rgba2 = red
        #     else:
        #         rgba1 = rgba_random
        #         rgba2 = rgba_random
        #     ic += [
        #         {
        #         "type": "sphere",
        #         "size": [0.004],
        #         "rgba": rgba1,
        #         "name": "site{}".format(i * sample_size + j),
        #         }
        #         for j in range(sample_size//2)
        #     ]
        #     ic += [
        #         {
        #         "type": "sphere",
        #         "size": [0.004],
        #         "rgba": rgba2,
        #         "name": "site{}".format(i * sample_size + sample_size//2 + j),
        #         }
        #         for j in range(sample_size//2)
        #     ]

        env.env = VisualizationWrapper(env.env, indicator_configs=ic)
        env.env.reset()
        env.env.set_visualization_setting('grippers', True)
        # from IPython import embed; embed()

    # load trained up model
    # eva = eval.Evaluator('yanweiw/robosuite/o4vfobq9')
    # eva = eval.Evaluator('yanweiw/robosuite/9oxfdli6')
    eva = eval.Evaluator(args.run_path)
    eva.load_model(epoch_num=args.epoch, root_dir=args.weight_dir)      

    # loop to visualize each trajectory
    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        data_save_path = None
        if args.gen_data_dir is not None:
            data_save_path = os.path.join(args.gen_data_dir, ep)
            os.makedirs(data_save_path, exist_ok=True)        
        
        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)], 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])

        # if is_robosuite_env:
        #     initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        orig_pos = f["data/{}/robot0_eef_pos".format(ep)][()] # [()] turn h5py dataset into numpy array
        gripper = f["data/{}/robot0_gripper_qpos".format(ep)][()] 
        gripper_state = ((gripper[:, 0] - gripper[:, 1]) > 0.06).astype(np.float32).reshape(-1, 1)
        # can_pos = f["data/{}/Can_pos".format(ep)][()]
        # mode_pred_states = np.hstack((can_pos-orig_pos, gripper, can_pos))
        obj_pos = f["data/{}/cube_pos".format(ep)][()]
        mode_pred_states = np.hstack((obj_pos-orig_pos, gripper, obj_pos))

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

            # supply eef pos
            # orig_pos = f["data/{}/obs/robot0_eef_pos".format(ep)][()] # [()] turn h5py dataset into numpy array
            eef_pos = perturb_traj(orig_pos, pert_range=0.2)
            # supply eef quat 
            eef_quat = f["data/{}/obs/robot0_eef_quat".format(ep)][()]
            # actions = np.hstack((eef_pos, eef_quat, actions[:, [-1]])) # append gripper action
            actions = np.hstack((eef_pos, actions[:, [-1]])) # append gripper action


        # from IPython import embed; embed()

        dict_of_obs, success = playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, orig_pos=orig_pos, actions=actions, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            demo_idx=ind,
            sample_size=sample_size,
            data_save_path=data_save_path,
            mode_pred_states=mode_pred_states,
            model=eva,
            guess_idx=args.guess_idx,
        )

        # from IPython import embed; embed()
        if args.gen_data_dir is not None:
            dict_of_obs['success'] = success
            # dict_of_obs['env_args'] = f['data'].attrs['env_args']
            with open(os.path.join(data_save_path, "env_args.txt"), "w") as outfile:
                outfile.write(f['data'].attrs['env_args'])
        
            np.savez(os.path.join(data_save_path, 'obs.npz'), **dict_of_obs)

    f.close()
    if write_video:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--gen_data_dir",
        type=str,
        default=None,
        help="(optional) path to directory where generated data is stored",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )
    # number of visualization sites
    parser.add_argument(
        "--ic", 
        type=int,
        default=100,
        help="(optional) number of visualization sites",
    )
    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use_obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use_actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--guess_idx",
        type=int,
        default=0,
        help="index of the guess to visualize",
    )

    # run_path
    parser.add_argument(
        "--run_path",
        type=str,
        default=None,
        help="path to the wandb run directory",
    )

    # epoch
    parser.add_argument(
        "--epoch",
        type=int,
        default=50000,
        help="epoch to load the model",
    )

    # weight_dir
    parser.add_argument(
        "--weight_dir",
        type=str,
        default='/home/felixw/mode_learning/weights',
        help="path to the base weight directory",
    )

    # use only succ demos
    parser.add_argument(
        "--succ",   
        action='store_true',
        help="use only successful demos",
    )

    # use only fail demos
    parser.add_argument(
        "--fail",
        action='store_true',
        help="use only failed demos",
    )

    args = parser.parse_args()
    playback_dataset(args)
