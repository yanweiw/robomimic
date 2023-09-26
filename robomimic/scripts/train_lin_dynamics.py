"""
Script that loads a mode classifer and a set of demonstrations and learns a policy
with a single stable point attractor per mode

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
from scipy.spatial.transform import Rotation as R
import random
import copy

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType
from robomimic.algo import register_algo_factory_func, PolicyAlgo, RolloutPolicy
import torch.nn as nn

from robosuite.wrappers import VisualizationWrapper
import sys 
sys.path.append('../../../mode_learning')
import eval 
import torch
import cv2
import copy

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



# MH custom class trains 'n' BC networks where each corresponds to a specific mode
class LINATTRACT(PolicyAlgo):
    """
    Point attractor dynamics that 
    """

    def __init__(self,mode_ends,env_meta,eva):
        self.env_meta = env_meta
        self.eva = eva
        self.mode_ends = mode_ends
        self.nets = nn.ModuleDict() # not used, just so it works with robomimic
        self.device = 'cpu'
        self.num_modes = self.eva.net.num_modes
        self.num_actions = 7 # TODO: not fixed
        self.k = 300
        self.b = 2.0*np.sqrt(self.k) # critically damped
        self.dt = 1/20.0 # 20 Hz control freq in robomimic?
        self.x_ddot = np.zeros((self.num_actions,))
        self.x_dot = np.zeros((self.num_actions,)) # TODO: could switch to actual velocity
        self.gripper_state = 1.0

        self._create_attractors(self.mode_ends)
        
    def _create_attractors(self,mode_ends):
        # action space is pos, rot(rpy), gripper action
        self.attractors = []
        self.relative_attractors = []

        for ii in range(self.num_modes):
            pos = np.mean(mode_ends[ii]["robot0_eef_pos"],axis=0)
            rel_pos = np.mean(mode_ends[ii]["robot0_eef_pos"]-np.array(mode_ends[ii]["object"])[:,0:3],axis=0)
            quat = np.mean(mode_ends[ii]["robot0_eef_quat"],axis=0) # TODO: shouldn't take mean :)
            rpy = R.from_quat(quat).as_euler('zxy')
            gripper_q = np.mean(mode_ends[ii]["robot0_gripper_qpos"],axis=0)
            gripper_state = int(gripper_q[0] - gripper_q[1] > 0.06)
            attractor_tmp = np.concatenate((pos.flatten(),rpy.flatten(),np.array(gripper_state).reshape((1,))))
            rel_attractor_tmp = np.concatenate((rel_pos.flatten(),rpy.flatten(),np.array(gripper_state).reshape((1,))))
            self.attractors.append(attractor_tmp)
            self.relative_attractors.append(rel_attractor_tmp)

    def _compute_control(self,obs_dict,mode):

        rpy = R.from_quat(obs_dict["robot0_eef_quat"].detach().cpu().numpy().flatten()).as_euler('zxy')
        pos = obs_dict["robot0_eef_pos"].detach().cpu().numpy().flatten()
        obj_pos = obs_dict["object"].detach().cpu().numpy().flatten()[0:3]
        gripper =  obs_dict["robot0_gripper_qpos"].detach().cpu().numpy().flatten()
        # gripper_state = int(gripper[0] - gripper[1] > 0.02)
        print(self.gripper_state)
        
        x = np.concatenate((pos,rpy,np.array(self.gripper_state).reshape((1,))))
        xd = self.attractors[mode]

        if mode==0 and True: # env TODO: update this
            xd = copy.deepcopy(self.relative_attractors[mode])
            xd[0:3] = xd[0:3] + obj_pos
            # xd[0:3] = obj_pos

        x_cmd = np.zeros((7,))
        # # dynamical system version
        # for ii in range(self.num_actions):
        #     self.x_ddot[ii] = -self.b*self.x_dot[ii] + self.k*(xd[ii]-x[ii])
        #     self.x_dot[ii] = self.x_dot[ii] + self.x_ddot[ii]*self.dt
        #     x[ii] = x[ii] + self.x_dot[ii]*self.dt
            # self.x_cmd[ii] = self.x
        # clipped delta version (go toward goal)
        # self.gripper_state = x[-1]
        # print("acc",self.x_ddot[2])
        # print("vel",self.x_dot[2])
        # self.x_dot[3:6] = 0.0 # TODO: remove
        # print("mode ",mode," ",x[-1],xd[-1])

        delta = xd - x
        self.x_dot = 4.0*delta
        self.x_dot[3:6] = 0

        # gripper stuff
        tmp = input()
        if 'o' in tmp:
            self.gripper_state = -1
        if 'c' in tmp:
            self.gripper_state = 1
            
        # if x[-1] > 0.01: # TODO: super hacky
        #     x[-1] = -1
        # else:
        #     x[-1] = 1
        print(self.gripper_state)
        self.x_dot[6] = self.gripper_state # gripper is an absolute action
        print("mode ",mode)
        # print(self.x_dot)
        # print(np.round(xd,2))
        # print(np.round(x,2))
        return self.x_dot

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        # Estimate Mode
        eef_pos_tmp = obs_dict["robot0_eef_pos"].detach().cpu().numpy().flatten()
        can_pos_tmp = obs_dict["object"].detach().cpu().numpy().flatten()[0:3]
        gripper =  obs_dict["robot0_gripper_qpos"].detach().cpu().numpy().flatten()
        mode_pred_states = np.concatenate((can_pos_tmp-eef_pos_tmp,gripper,can_pos_tmp))
        guess_idx=0
        mode_pred_states = torch.tensor(mode_pred_states, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mode, mode_log, mode_logits = self.eva.net.pred_mode(mode_pred_states.cuda())
        traj_len = 1
        mode = mode.reshape(-1, traj_len, self.eva.net.num_guess, self.eva.net.num_modes)[:, :, guess_idx, :]
        mode_idx = torch.argmax(mode, dim=-1).detach().cpu().numpy()
        mode_idx = mode_idx[0][0]

        # Compute control law
        # print(torch.from_numpy(self._compute_control(obs_dict,mode_idx).reshape((1,len(self._compute_control(obs_dict,mode_idx))))))
        return torch.from_numpy(self._compute_control(obs_dict,mode_idx).reshape((1,len(self._compute_control(obs_dict,mode_idx)))))
        

def learn_linear_system(args):

    # if args.use_obs:
    #     assert write_video, "playback with observations can only write to video"
    #     assert not args.use_actions, "playback with observations is offline and does not support action playback"

    # # create environment only if not playing back with observations
    # if not args.use_obs:
    #     # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
    #     # for playback since observations are unused. Pass a dummy spec here.
    #     dummy_spec = dict(
    #         obs=dict(
    #                 low_dim=["robot0_eef_pos"],
    #                 rgb=[],
    #             ),
    #     )
    #     ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    #     # from IPython import embed; embed()
    #     env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    #     # directly control ee pose
    #     env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    #     env_meta['env_kwargs']['controller_configs']['control_ori'] = False
    #     env_meta['env_kwargs']['controller_configs']['kp'] = 1000
    #     env_meta['env_kwargs']['controller_configs']['kp_limits'] = [0, 1000]
    #     env_meta['env_kwargs']['controller_configs']['output_max'] = [2, 2, 2, 1, 1, 1, 1] # these values are just placeholders
    #     env_meta['env_kwargs']['controller_configs']['output_min'] = [-2, -2, -2, -1, -1, -1, -1]        
    #     env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

    #     # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    #     is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env_type = EnvUtils.get_env_type(env_meta=env_meta)

    # print(f["data/demo_0/obs/object"].keys())
    # input()
    # from IPython import embed; embed()
    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())


    # load trained up model
    eva = eval.Evaluator(args.run_path)
    eva.load_model(epoch_num=args.epoch, root_dir=args.weight_dir) 
    num_classes = eva.net.num_modes    

    # TODO: get the mean ending values and state conversions for each mode
    # TODO: create the attractor and create a policy around it
    # TODO: get things running end to end with run_trained_agent
    # TODO: circle back and make work better/augment as necessary

    # higher level:
    # TODO: need to be able to train from the perturbed dataset (dataset needs some restructuring to made hd5f from robomimic)
    # TODO: script to run a suite of tests

    # MHLOOP
    # loop to visualize each trajectory
    mode_ends = [None] * num_classes # number of modes
    for ind in range(len(demos)):
        
        # extract demo state data
        ep = demos[ind]
        orig_pos = f["data/{}/obs/robot0_eef_pos".format(ep)][()] # [()] turn h5py dataset into numpy array
        orig_rot = f["data/{}/obs/robot0_eef_quat".format(ep)][()] # [()] turn h5py dataset into numpy array
        gripper = f["data/{}/obs/robot0_gripper_qpos".format(ep)][()] 
        gripper_state = ((gripper[:, 0] - gripper[:, 1]) > 0.06).astype(np.float32).reshape(-1, 1)
        obj = f["data/{}/obs/object".format(ep)][()]
        can_pos = obj[:,0:3]

        mode_pred_states = np.hstack((can_pos-orig_pos, gripper, can_pos))

        # estimate modes
        guess_idx=0
        mode_pred_states = torch.tensor(mode_pred_states, dtype=torch.float32).unsqueeze(0)
        traj_len = mode_pred_states.shape[1]
        with torch.no_grad():
            mode, mode_log, mode_logits = eva.net.pred_mode(mode_pred_states.cuda())
        mode = mode.reshape(-1, traj_len, eva.net.num_guess, eva.net.num_modes)[:, :, guess_idx, :]
        mode_idx = torch.argmax(mode, dim=-1)
        mode_idx = mode_idx.detach().cpu().numpy()[0] # indexing to remove batch dim

        # identify and store valid mode transitions
        mode_changes = np.where(np.diff(mode_idx,prepend=np.nan))[0][1:]
        last_mode = mode_changes # one sample before the change TODO: don't subtract one to encourage tx
        # print(mode_idx)
        # print(last_mode)
        # input()

        for idtmp in last_mode:
            # store ee, gripper, and can
            # can_eef_pos = np.array(can_pos[idtmp]-orig_pos[idtmp])
            # des_action = np.concatenate((can_eef_pos,gripper_state[idtmp],can_pos[idtmp]))
            # print("Mode: ",mode_idx[idtmp]," action:",des_action)
            if mode_ends[mode_idx[idtmp]] is None:
                mode_ends[mode_idx[idtmp]] = dict()

            # append all possible interesting action information
            if "robot0_eef_pos" not in mode_ends[mode_idx[idtmp]].keys():
                mode_ends[mode_idx[idtmp]]["robot0_eef_pos"] = []
                mode_ends[mode_idx[idtmp]]["robot0_eef_quat"] = []
                mode_ends[mode_idx[idtmp]]["robot0_gripper_qpos"] = []
                mode_ends[mode_idx[idtmp]]["object"] = []
            mode_ends[mode_idx[idtmp]]["robot0_eef_pos"].append(orig_pos[idtmp])
            mode_ends[mode_idx[idtmp]]["robot0_eef_quat"].append(orig_rot[idtmp])
            mode_ends[mode_idx[idtmp]]["robot0_gripper_qpos"].append(gripper[idtmp])
            mode_ends[mode_idx[idtmp]]["object"].append(obj[idtmp])

        # policy looks at current mode and pulls toward
        
    # print("MH assessing mode ends")
    # print(np.shape(mode_ends))
    # print(np.shape(mode_ends[0]["robot0_eef_pos"]))
    # print(np.shape(mode_ends[0]["object"]))
    # print(np.shape(mode_ends[1]["object"]))
    # print(np.mean(mode_ends[0]["robot0_eef_pos"]-np.array(mode_ends[0]["object"])[:,0:3],axis=0))

    linpolicy = LINATTRACT(mode_ends,env_meta,eva)
    rolloutpolicy = RolloutPolicy(linpolicy)
    assert args.policy_path is not None
    torch.save(rolloutpolicy,args.policy_path)

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

    # run_path
    parser.add_argument(
        "--run_path",
        type=str,
        default=None,
        help="path to the wandb run directory",
    )

    # policy_path
    parser.add_argument(
        "--policy_path",
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

    args = parser.parse_args()
    learn_linear_system(args)
