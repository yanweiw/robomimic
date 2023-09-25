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
import matplotlib.pyplot as plt
import numpy as np
import time

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

# class MultiDataStreamPlotter:
#     def __init__(self, num_streams, stream_labels):
#         self.num_streams = num_streams
#         self.stream_labels = stream_labels
#         self.data_streams = [[] for _ in range(num_streams)]
#         self.lines = []

#         self.setup_plot()

#     def setup_plot(self):
#         self.fig, self.ax = plt.subplots()
#         self.ax.set_xlabel('X-axis')
#         self.ax.set_ylabel('Y-axis')
#         self.ax.set_title('Multiple Data Streams Plot')
#         self.ax.legend(self.stream_labels, loc='upper right')

#         # Initialize lines for each data stream
#         self.lines = [self.ax.plot([], [], label=label)[0] for label in self.stream_labels]

#     def update_plot(self, new_data):
#         for i, new_y in enumerate(new_data):
#             self.data_streams[i].append(new_y)
#             self.lines[i].set_data(range(len(self.data_streams[i])), self.data_streams[i])

#         self.ax.relim()
#         self.ax.autoscale_view()
#         plt.draw()

#     def show_plot(self):
#         plt.show()

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
    # obs_data = None,
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

    action_playback = False #(actions is not None)

    # downsample the trajectory to a fixed size for visualization
    assert sample_size is not None
    orig_idx = np.array(range(states.shape[0]))
    sampled_idx = downsample_array(orig_idx, sample_size)
    
    # plot the original ee positions as a reference
    ic_list = []


    # render the simulation
    for i in range(len(states)):
        env.reset_to({"states" : states[i]})

        if i in sampled_idx:
            in_demo_idx = np.where(sampled_idx == i)[0][0]
            ic_idx = demo_idx * sample_size + in_demo_idx
            if action_playback:
                ic_idx += sample_size//2
            
            env.env.set_indicator_pos("site{}".format(ic_idx), env.env._get_observations(force_update=True)["robot0_eef_pos"])
            ic_list.append("site{}".format(ic_idx))

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
        env_meta['env_kwargs']['controller_configs']['kp'] = 150
        # env_meta['env_kwargs']['controller_configs']['kp_limits'] = [0, 1000]
        # env_meta['env_kwargs']['controller_configs']['output_max'] = [2, 2, 2, 1, 1, 1, ] # these values are just placeholders
        # env_meta['env_kwargs']['controller_configs']['output_min'] = [-2, -2, -2, -1, -1, -1]        
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

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

    if args.et:
        demos = [demo for demo in demos if 'et' in demo]
    elif args.pg:
        demos = [demo for demo in demos if 'pg' in demo and 'et' not in demo]
    elif args.pe:
        demos = [demo for demo in demos if 'pe' in demo and 'et' not in demo and 'pg' not in demo]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[args.n:(args.n+30)]
    elif args.demo is not None:
        demos = [d for d in demos if args.demo in d]
        if len(demos) == 1:
            demos = [demos[0]] * 3 # repeat the demo 3 times

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    if not args.use_obs:
        sample_size = args.ic
        if args.use_actions:
            sample_size = sample_size * 2
        ic = []
        for i in range(len(demos)):
            rgba_random = np.random.uniform(0, 1, 3).tolist() + [0.5]
            blue = [0, 0, 1, 1] 
            red = [1, 0, 0, 1]
            if args.use_actions:
                rgba1 = blue
                rgba2 = red
            else:
                rgba1 = rgba_random
                rgba2 = rgba_random
            ic += [
                {
                "type": "sphere",
                "size": [0.004],
                "rgba": rgba1,
                "name": "site{}".format(i * sample_size + j),
                }
                for j in range(sample_size//2)
            ]
            ic += [
                {
                "type": "sphere",
                "size": [0.004],
                "rgba": rgba2,
                "name": "site{}".format(i * sample_size + sample_size//2 + j),
                }
                for j in range(sample_size//2)
            ]

        env.env = VisualizationWrapper(env.env, indicator_configs=ic)
        env.env.reset()
        env.env.set_visualization_setting('grippers', True)

    # orig_actions = f["data/{}/actions".format(demos[0])][()]
    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))
        orig_pos = None
        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        # obs_data = {}
        # obs_data['nut_pos'] = f['data'][ep]['SquareNut_pos'][:]
        # obs_data['nut_quat'] = f['data'][ep]['SquareNut_quat'][:]
        # obs_data['nut2eef_pos'] = f['data'][ep]['SquareNut_to_robot0_eef_pos'][:]
        # obs_data['nut2eef_quat'] = f['data'][ep]['SquareNut_to_robot0_eef_quat'][:]

        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, orig_pos=orig_pos, actions=None, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            demo_idx=ind,
            sample_size=sample_size,
            data_save_path=None,
            # obs_data=obs_data,
        )

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

    # Replay a particular demonstration
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="(optional) replay a particular demonstration",
    )

    # succ tag
    parser.add_argument(
        "--succ",
        action='store_true',
        help="only playback successful demonstrations",
    )

    # fail tag
    parser.add_argument(
        "--fail",
        action='store_true',
        help="only playback failed demonstrations",
    )

    # et tag
    parser.add_argument(
        "--et",
        action='store_true',
        help="only playback et demonstrations",
    )

    # pe tag
    parser.add_argument(
        "--pe",
        action='store_true',
        help="only playback pe demonstrations",
    )

    # pg tag
    parser.add_argument(
        "--pg",
        action='store_true',
        help="only playback pg demonstrations",
    )

    args = parser.parse_args()
    playback_dataset(args)
