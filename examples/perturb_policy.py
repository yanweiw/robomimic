import argparse
import json
import h5py
import imageio
import numpy as np
import os
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

import urllib.request
from IPython import embed
import time

class Simulate():
    def __init__(self, policy, env, render=False, video_writer=None, video_skip=5, camera_names=None) -> None:
        self.policy = policy
        self.env = env
        self.if_render = render
        self.video_writer = video_writer
        self.video_skip = video_skip
        self.camera_names = camera_names

    def rollout(self, horizon, parser):
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
            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.
        Returns:
            stats (dict): some statistics for the rollout - such as return, horizon, and task success
        """
        assert isinstance(self.env, EnvBase)
        assert isinstance(self.policy, RolloutPolicy)
        assert not (self.if_render and (self.video_writer is not None))

        self.policy.start_episode()
        obs = self.env.reset()
        state_dict = self.env.get_state()

        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = self.env.reset_to(state_dict)

        results = {}
        video_count = 0  # video frame counter
        total_reward = 0.
        random_idx = np.random.randint(0, 7)
        try:
            for step_i in range(horizon):

                # get action from policy
                act = self.policy(ob=obs)
                # time.sleep(0.05)
                if step_i >= args.perturb_start and step_i < args.perturb_start+10:
                    print(act)
                    act[args.perturb_idx] += args.perturb_mag
                    print(act)
                    print('\n')

                # play action
                next_obs, r, done, _ = self.env.step(act)

                # compute reward
                total_reward += r
                success = self.env.is_success()["task"]

                # visualization
                if self.if_render:
                    self.env.render(mode="human", camera_name=self.camera_names[0])
                if self.video_writer is not None:
                    if video_count % self.video_skip == 0:
                        video_img = []
                        for cam_name in self.camera_names:
                            video_img.append(self.env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                        video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                        self.video_writer.append_data(video_img)
                    video_count += 1

                # break if done or if success
                if done or success:
                    break

                # update for next iter
                obs = deepcopy(next_obs)
                state_dict = self.env.get_state()

        except self.env.rollout_exceptions as e:
            print("WARNING: got rollout exception {}".format(e))

        stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

        return stats

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--perturb_idx', type=int, default=0)
    parser.add_argument('-m', '--perturb_mag', type=float, default=0)
    parser.add_argument('-s', '--perturb_start', type=int, default=0)
    args = parser.parse_args()

    ckpt_path = ckpt_path = 'bc_result/models/model_epoch_300_Lift_success_1.0.pth'
    assert os.path.exists(ckpt_path)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        render=True, # we won't do on-screen rendering in the notebook
        render_offscreen=False, # render to RGB images for video
        verbose=True,
    )

    # np.random.seed(0)
    # torch.manual_seed(0)

    sim = Simulate(policy, env, render=True, camera_names=["agentview"])

    while True:
        stats = sim.rollout(200, parser)
        embed()
        time.sleep(0.0)
    