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


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


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
    log_data = False,
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
    # if action_playback:
    #     assert states.shape[0] == actions.shape[0]

    # downsample the trajectory to a fixed size for visualization
    assert sample_size is not None
    orig_idx = np.array(range(actions.shape[0]))
    if action_playback: 
        # save the first half of sites for original data; and the second half for perturbed data
        sampled_idx = downsample_array(orig_idx, sample_size//2)
    else:
        sampled_idx = downsample_array(orig_idx, sample_size)
    
    # plot the original ee positions as a reference
    ic_list = []
    if action_playback:
        # env.reset() # load the initial state (it will close the simulation window and re-open it)
        # env.reset_to(initial_state)        
        # env.reset_to({"states": states[0]})       
        for i, ee_pos in enumerate(orig_pos):
            if i in sampled_idx:
                in_demo_idx = np.where(sampled_idx == i)[0][0]
                ic_idx = in_demo_idx
                env.env.set_indicator_pos("site{}".format(ic_idx), ee_pos)
                # print("setting indiciator sites{}".format(ic_idx))
                ic_list.append("site{}".format(ic_idx))
                # env.env.sim.forward()
        env.reset_to({"states": states[0]})

    # accumulate data for each step
    keys = list(env.env.observation_spec().keys())
    keys.append('states')
    keys.append('actions')
    dict_of_arrays = {key: [] for key in keys}
    # render the simulation
    for i in range(len(actions)):
        if not action_playback:
            env.reset_to({"states" : states[i]})
        else:
            env.step(actions[i])
            if log_data:
                # save the data for each step
                dict_of_arrays['states'].append(env.get_state()["states"])
                dict_of_arrays['actions'].append(actions[i])
                obs = env.env.observation_spec()
                for key in obs.keys():
                    dict_of_arrays[key].append(obs[key])

        if i in sampled_idx:
            in_demo_idx = np.where(sampled_idx == i)[0][0]
            ic_idx = in_demo_idx
            if action_playback:
                ic_idx += sample_size//2
            
            env.env.set_indicator_pos("site{}".format(ic_idx), env.env._get_observations(force_update=True)["robot0_eef_pos"])
            # print("setting indiciator sites{}".format(ic_idx))
            ic_list.append("site{}".format(ic_idx))
            # env.env.sim.forward()

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        if first:
            break

    # remove the indicator sites to reduce clutter
    if action_playback:
        for ic in ic_list:
            env.env.set_indicator_pos(ic, [0, 0, 0])

    if log_data is not None:
        dict_of_arrays = {key: np.vstack(dict_of_arrays[key]) for key in dict_of_arrays.keys()}
        return dict_of_arrays, env.get_reward()
    else:
        return None, None


def perturb_traj(actions, pert_range=0.1, perturb_grasp=False, final_non_perturb_len=14):
    # orig actions (traj_len, 3)
    assert actions.shape[1] == 4
    orig_ee = actions[:, :-1]
    gripper_pos = actions[:, [-1]]
    min_perturb_len = 20
    assert len(actions) > min_perturb_len + final_non_perturb_len # last 14 gripper actions are open in demos
    perturbed_ee = orig_ee.copy()
    if not perturb_grasp: 
        impulse_start = random.randint(0, len(orig_ee)-min_perturb_len-final_non_perturb_len)
        impulse_end = random.randint(impulse_start+min_perturb_len, len(orig_ee)-final_non_perturb_len)
        impulse_mean = (impulse_start + impulse_end)//2
        impulse_mean_action = orig_ee[impulse_mean]
        impulse_targets = []
        for curr in impulse_mean_action: # 3d for ee_pos
            target = random.uniform(curr-pert_range, curr+pert_range)
            # if target < -1: target = -1
            # if target > 1: target = 1
            impulse_targets.append(target)
        # impulse_target_x = random.uniform(-8, 8)
        # impulse_target_y = random.uniform(-8, 8)
        max_relative_dist = 5 # np.exp(-5) ~= 0.006

        kernel = np.exp(-max_relative_dist * (np.array(range(len(orig_ee))) - impulse_mean)**2 / ((impulse_start-impulse_mean)**2))
        for i in range(orig_ee.shape[1]):
            perturbed_ee[:, i] += (impulse_targets[i] - perturbed_ee[:, i]) * kernel

    # appending gripper actions
    perturbed_gripper = gripper_pos.copy()
    if perturb_grasp: 
        max_gripper_pertrub_len = 20
        gripper_perturb_len = random.randint(10, max_gripper_pertrub_len)
        gripper_perturb_start = random.randint(0, len(perturbed_gripper)-gripper_perturb_len-final_non_perturb_len) # last 14 gripper actions are open in demos
        perturbed_gripper[gripper_perturb_start:gripper_perturb_start+gripper_perturb_len] = -1 # flip gripper actions; -1 is open and 1 is close

    perturbed = np.hstack((perturbed_ee, perturbed_gripper)) # append gripper action

    return perturbed

def pulse_train(actions, pert_mag=0.1):
    # orig actions (traj_len, 3)
    assert actions.shape[1] == 4
    perturb_len = 10
    assert len(actions) > perturb_len * 3
    max_relative_dist = 5 # np.exp(-5) ~= 0.006
    perturbed_actions_list = []
    for impulse_start in np.linspace(perturb_len, len(actions)-2*perturb_len, 10, dtype=int):
        impulse_mean = impulse_start + perturb_len//2
        random_direction = np.random.rand(3)
        normalized_direction = random_direction / np.linalg.norm(random_direction)
        pert_vec = normalized_direction * pert_mag
        kernel = np.exp(-max_relative_dist * (np.array(range(len(actions))) - impulse_mean)**2 / ((impulse_start-impulse_mean)**2))
        perturbed_actions = actions.copy()
        perturbed_actions[:, :3] +=  kernel.reshape(-1, 1) * pert_vec.reshape(1, -1)
        perturbed_actions_list.append(perturbed_actions)
        # adding gripper perturbations
        perturbed_actions = actions.copy()
        perturbed_actions[impulse_start:impulse_start+perturb_len, -1] *= -1
        perturbed_actions_list.append(perturbed_actions)

    return perturbed_actions_list

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

    # create environment only if not playing back with observations
    # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
    # for playback since observations are unused. Pass a dummy spec here.
    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    # directly control ee pose
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    env_meta['env_kwargs']['controller_configs']['control_ori'] = False
    env_meta['env_kwargs']['controller_configs']['kp'] = 1000
    env_meta['env_kwargs']['controller_configs']['kp_limits'] = [0, 1000]
    env_meta['env_kwargs']['controller_configs']['output_max'] = [2, 2, 2, 1, 1, 1, 1] # these values are just placeholders
    env_meta['env_kwargs']['controller_configs']['output_min'] = [-2, -2, -2, -1, -1, -1, -1]        
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

    f = h5py.File(args.dataset, "r")
    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # change the number of demonstrations to playback
    if args.n is not None:
        new_demos = []
        while len(new_demos) < args.n:
            new_demos.extend(demos)
        demos = new_demos[:args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    sample_size = args.ic * 2 # first half for original data, second half for perturbed data
    ic = []
    blue = [0, 0, 1, 1] 
    red = [1, 0, 0, 1]
    ic += [
        {
        "type": "sphere",
        "size": [0.004],
        "rgba": blue,
        "name": "site{}".format(j),
        }
        for j in range(sample_size//2)
    ]
    ic += [
        {
        "type": "sphere",
        "size": [0.004],
        "rgba": red,
        "name": "site{}".format(sample_size//2 + j),
        }
        for j in range(sample_size//2)
    ]

    env.env = VisualizationWrapper(env.env, indicator_configs=ic)
    env.env.reset()
    env.env.set_visualization_setting('grippers', True)

    for ind, ep in enumerate(demos):
        # ep is of format 'demo_0', 'demo_1', etc. Let's pad the number with zeros
        print("Playing back episode: {}".format(ep))      

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])

        # supply actions if using open-loop action playback
        actions = f["data/{}/actions".format(ep)][()]

        # supply eef pos
        orig_ee_pos = f["data/{}/obs/robot0_eef_pos".format(ep)][()] # [()] turn h5py dataset into numpy array
        gripper_pos = actions[:, [-1]]
        actions = np.hstack((orig_ee_pos, gripper_pos)) # append gripper action
        perturb_type = random.choice(['pe', 'pg']) # pe: perturb ee; pg: perturb gripper
        actions = perturb_traj(actions, pert_range=args.pert_range, perturb_grasp=(perturb_type=='pg'), final_non_perturb_len=args.non_pert)
        # perturbed_actions_list = pulse_train(actions, pert_mag=args.pert_range)

        for pert_idx, perturbed_actions in enumerate([actions]):
            assert pert_idx == 0 # only one perturbation for now, otherwise following perturbations will overwrite this one
            dict_of_obs, success = playback_trajectory_with_env(
                env=env, 
                initial_state=initial_state, 
                states=states, orig_pos=orig_ee_pos, actions=perturbed_actions, 
                render=args.render, 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                camera_names=args.render_image_names,
                first=args.first,
                demo_idx=ind,
                sample_size=sample_size,
                log_data=True,
            )

            if args.gen_data_dir is not None:
                if success:
                    data_save_path = os.path.join(args.gen_data_dir, str(ind).zfill(4) + '_' + perturb_type + '_00' + '_succ')
                else:
                    data_save_path = os.path.join(args.gen_data_dir, str(ind).zfill(4) + '_' + perturb_type + '_00' + '_fail')
                os.makedirs(data_save_path, exist_ok=True)              
                dict_of_obs['success'] = success
                with open(os.path.join(data_save_path, "env_args.txt"), "w") as outfile:
                    outfile.write(f['data'].attrs['env_args'])        
                np.savez(os.path.join(data_save_path, 'obs.npz'), **dict_of_obs)

        if success:
            # 1. perturb ending location
            early_terminate_idx = random.randint(int(len(actions)*0.9), len(actions))
            new_actions = actions.copy()[:early_terminate_idx]
            dict_of_obs, success = playback_trajectory_with_env(
                env=env, 
                initial_state=initial_state, 
                states=states, orig_pos=orig_ee_pos, actions=new_actions, 
                render=args.render, 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                camera_names=args.render_image_names,
                first=args.first,
                demo_idx=ind,
                sample_size=sample_size,
                log_data=True,
            )
            if args.gen_data_dir is not None:
                if success:
                    data_save_path = os.path.join(args.gen_data_dir, str(ind).zfill(4) + '_' + perturb_type + '_et' + '_succ')
                else:
                    data_save_path = os.path.join(args.gen_data_dir, str(ind).zfill(4) + '_' + perturb_type + '_et' + '_fail')
                os.makedirs(data_save_path, exist_ok=True)  
                dict_of_obs['success'] = success
                with open(os.path.join(data_save_path, "env_args.txt"), "w") as outfile:
                    outfile.write(f['data'].attrs['env_args'])
                np.savez(os.path.join(data_save_path, 'obs.npz'), **dict_of_obs)

            if perturb_type=='pe':
                # 2. perturb grasp
                new_actions = perturb_traj(actions, pert_range=args.pert_range, perturb_grasp=True, final_non_perturb_len=args.non_pert)
                dict_of_obs, success = playback_trajectory_with_env(
                    env=env, 
                    initial_state=initial_state, 
                    states=states, orig_pos=orig_ee_pos, actions=new_actions, 
                    render=args.render, 
                    video_writer=video_writer, 
                    video_skip=args.video_skip,
                    camera_names=args.render_image_names,
                    first=args.first,
                    demo_idx=ind,
                    sample_size=sample_size,
                    log_data=True,
                )
                if args.gen_data_dir is not None:
                    if success:
                        data_save_path = os.path.join(args.gen_data_dir, str(ind).zfill(4) + '_pe_pg' + '_succ')
                    else:
                        data_save_path = os.path.join(args.gen_data_dir, str(ind).zfill(4) + '_pe_pg' + '_fail')
                    os.makedirs(data_save_path, exist_ok=True)  
                    dict_of_obs['success'] = success
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
        default=200,
        help="(optional) number of visualization sites",
    )
    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
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

    # perturbation range
    parser.add_argument(
        "--pert_range",
        type=float,
        default=0.2,
        help="perturbation range",
    )

    # number of non-perturbed actions at the end of each trajectory
    parser.add_argument(
        "--non_pert",
        type=int,
        default=14,
        help="number of non-perturbed actions at the end of each trajectory",
    )

    args = parser.parse_args()
    playback_dataset(args)
