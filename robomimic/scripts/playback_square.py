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
import robosuite.utils.transform_utils as T


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
    data_save_path = None,
    site_obs = None,
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
    if data_save_path is not None:
        action_playback = False # just copy the original actions

    # downsample the trajectory to a fixed size for visualization
    assert sample_size is not None
    orig_idx = np.array(range(states.shape[0]))
    # if action_playback: 
    #     # save the first half of sites for original data; and the second half for perturbed data
    #     sampled_idx = downsample_array(orig_idx, sample_size//2)
    # else:
    sampled_idx = downsample_array(orig_idx, sample_size)

    # plot the original ee positions as a reference
    ic_list = []
    # if action_playback:
    #     # get the orignal sequence of ee positions by playing back joint states
    #     for i, ee_pos in enumerate(orig_pos):
    #         if i in sampled_idx:
    #             in_demo_idx = np.where(sampled_idx == i)[0][0]
    #             ic_idx = demo_idx * sample_size + in_demo_idx
    #             env.env.set_indicator_pos("site{}".format(ic_idx), ee_pos)
    #             ic_list.append("site{}".format(ic_idx))
    #     env.reset_to({"states": states[0]})

    # # acculate data for each step
    # keys = list(env.env.observation_spec().keys())
    # keys.append('states')
    # keys.append('actions')
    # for site in ['peg_site', 'gripper0_grip_site', 'gripper0_left_ee_site', 'gripper0_right_ee_site', 'SquareNut_handle_site', 'SquareNut_center_site', 'SquareNut_side_site']:
    #     keys.append(site)
    # dict_of_arrays = {key: [] for key in keys}

    # render the simulation
    for i in range(len(states)):
        if not action_playback:
            env.reset_to({"states" : states[i]})
        else:
            env.step(actions[i])

        # if data_save_path is not None:
        #     # save the data for each step
        #     dict_of_arrays['states'].append(env.get_state()["states"])
        #     dict_of_arrays['actions'].append(actions[i])
        #     obs = env.env.observation_spec()
        #     for key in obs.keys():
        #         dict_of_arrays[key].append(obs[key])

        #     for site in ['peg_site', 'gripper0_grip_site', 'gripper0_left_ee_site', 'gripper0_right_ee_site', 'SquareNut_handle_site', 'SquareNut_center_site', 'SquareNut_side_site']:
        #         site_pos = env.env.sim.data.site_xpos[env.env.sim.model.site_name2id(site)]
        #         dict_of_arrays[site].append(site_pos)

            # plot relative distance changes
            # nut_to_eef_pos = env.env._get_observations(force_update=True)["SquareNut_to_robot0_eef_pos"]
            # # print([f"{value:.3f}" for value in nut_to_eef_pos])
            # nut_to_eef_quat = env.env._get_observations(force_update=True)["SquareNut_to_robot0_eef_quat"]
            # nut_to_eef_axis_angle = T.quat2axisangle(nut_to_eef_quat)
            # # print([f"{value:.3f}" for value in nut_to_eef_axis_angle])
            

        if i in sampled_idx:
            in_demo_idx = np.where(sampled_idx == i)[0][0]
            # ic_idx = demo_idx * sample_size + in_demo_idx
            # if action_playback:
                # ic_idx += sample_size//2
            
            # if data_save_path is not None:
            for j, site in enumerate(site_obs.keys()):
                ic_idx = j * sample_size + in_demo_idx
                # from IPython import embed; embed()
                env.env.set_indicator_pos("site{}".format(ic_idx), site_obs[site][i])
                # site_pos = env.env.sim.data.site_xpos[env.env.sim.model.site_name2id(site)]
                # env.env.set_indicator_pos("site{}".format(ic_idx), site_pos)
                ic_list.append("site{}".format(ic_idx))
            # env.env.set_indicator_pos("site{}".format(ic_idx), env.env._get_observations(force_update=True)["robot0_eef_pos"])
            # ic_list.append("site{}".format(ic_idx))

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
    # if action_playback:
    for ic in ic_list:
        env.env.set_indicator_pos(ic, [0, 0, 0])

    if data_save_path is not None:
        dict_of_arrays = {key: np.vstack(dict_of_arrays[key]) for key in dict_of_arrays.keys()}
        return dict_of_arrays, env.get_reward()
    else:
        return None, None


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

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        # directly control ee pose
        env_meta['env_kwargs']['controller_configs']['control_delta'] = True
        env_meta['env_kwargs']['controller_configs']['control_ori'] = True
        env_meta['env_kwargs']['control_freq'] = 20
        # env_meta['env_kwargs']['controller_configs']['uncouple_pos_ori'] = False # important to set orientation state
        env_meta['env_kwargs']['controller_configs']['kp'] = 150 # 150
        env_meta['env_kwargs']['controller_configs']['damping'] = 1 # 1
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

    f = h5py.File(args.dataset, "r")
    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[args.n:args.n+20]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    if not args.use_obs:
        sample_size = args.ic
        # if args.use_actions:
            # sample_size = sample_size * 2
        ic = []
        num_sites = 10
        for i in range(num_sites):
            rgba_random = np.random.uniform(0, 1, 3).tolist() + [0.5]
            # blue = [0, 0, 1, 1] 
            # red = [1, 0, 0, 1]
            # if args.use_actions:
                # rgba1 = blue
                # rgba2 = red
            # else:
                # rgba1 = rgba_random
                # rgba2 = rgba_random
            # ic += [
            #     {
            #     "type": "sphere",
            #     "size": [0.004],
            #     "rgba": rgba1,
            #     "name": "site{}".format(i * sample_size + j),
            #     }
            #     for j in range(sample_size//2)
            # ]
            # ic += [
            #     {
            #     "type": "sphere",
            #     "size": [0.004],
            #     "rgba": rgba2,
            #     "name": "site{}".format(i * sample_size + sample_size//2 + j),
            #     }
            #     for j in range(sample_size//2)
            # ]
            for j in range(sample_size):
                ic += [
                    {
                    "type": "sphere",
                    "size": [0.004],
                    "rgba": rgba_random,
                    "name": "site{}".format(i * sample_size + j),
                    }
                ]
        env.env = VisualizationWrapper(env.env, indicator_configs=ic)
        env.env.reset()
        env.env.set_visualization_setting('grippers', True)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        orig_pos = None
        data_save_path = None
        if args.gen_data_dir is not None:
            data_save_path = os.path.join(args.gen_data_dir, ep)
            os.makedirs(data_save_path, exist_ok=True)        

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        sites_obs = {}
        for site in ['peg_site', 'gripper0_grip_site', 'gripper0_left_ee_site', 'gripper0_right_ee_site', 'SquareNut_handle_site', 'SquareNut_center_site', 'SquareNut_side_site']:
            sites_obs[site] = f["data/{}/{}".format(ep, site)][()]

        # supply actions if using open-loop action playback
        actions = None
        # if args.use_actions:
        #     actions = f["data/{}/actions".format(ep)][()]
        #     if data_save_path is not None: # save original actions, otherwise, supply eef pos
        #         orig_pos = f["data/{}/obs/robot0_eef_pos".format(ep)][()] # [()] turn h5py dataset into numpy array
        #         # orig_quat = f["data/{}/obs/robot0_eef_quat".format(ep)][()]
        #         # eef_pos = perturb_traj(orig_pos, pert_range=0.2)
        #         # actions = np.hstack((eef_pos, actions[:, 3:6], actions[:, [-1]])) # append euler angle delta and gripper action
        #         actions = np.hstack((orig_pos, actions[:, 3:6],  actions[:, [-1]]))


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
            site_obs=sites_obs,
        )

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
    parser.add_argument( # only use this option to add sites to the perturbed trajectories without sites
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

    args = parser.parse_args()
    playback_dataset(args)
