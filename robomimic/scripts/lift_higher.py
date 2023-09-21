# Script to modify original demo of the lifting task

# - The original human demos only lifts the cube to a height of around 0.88, which exceeds [the original succ criteria of 0.04](https://github.com/yanweiw/robosuite/blob/eb01e1ffa46f1af0a3aa3ac363d5e63097a6cbcc/robosuite/environments/manipulation/lift.py#L428) but not [the new one of 0.44](https://github.com/yanweiw/robosuite/blob/e5a60dd073768509a4ffe81eeff33f3161af0be9/robosuite/environments/manipulation/lift.py#L428)
# - So I will manually append 10 actions of 0.010m in the z direction to the original data
#     - repeat the last state in “data/demo_*/obs/robot0_eef_pos” 40 times and change the z value
#     - repeat the last action in “data/demo_*/actions” 40 times (this only matter for gripper state)

import argparse
import h5py
import numpy as np


args = argparse.ArgumentParser()
args.add_argument("--demo_path", type=str, default="data/demo_0.hdf5")
args.add_argument("--output_path", type=str, default="data/demo_0_modified.hdf5")

args = args.parse_args()

out = h5py.File(args.output_path, "w")

repeat_num = 40
lift_speed = 0.01

with h5py.File(args.demo_path, "r+") as f:
    for ep in f['data'].keys():
        ee_pos = f["data/{}/obs/robot0_eef_pos".format(ep)][()]
        # curr_shape = ee_pos.shape
        new_entries = np.repeat(ee_pos[[-1]], repeat_num, axis=0)
        bias = np.random.uniform(-0.002, 0.002, size=2)
        for i in range(repeat_num):
            if i > 0:
                new_entries[i, :2] = new_entries[i-1, :2] + np.random.normal(bias, [0.001, 0.001])
            new_entries[i, 2] += lift_speed * (i + 1)
        new_ee_pos = np.vstack((ee_pos, new_entries))
        out.create_dataset("data/{}/obs/robot0_eef_pos".format(ep), data=new_ee_pos)
        # new_ee_pos = f.create_dataset("eef_pos_new", shape=(curr_shape[0] + 10, curr_shape[1]))
        # new_ee_pos[:curr_shape[0]] = ee_pos[:]
        # ee_pos.resize((curr_shape[0] + 10, curr_shape[1]))
        # new_ee_pos[curr_shape[0]:] = new_entries

        actions = f["data/{}/actions".format(ep)]
        # curr_shape = actions.shape
        new_entries = np.repeat(actions[[-1]], repeat_num, axis=0)
        new_actions = np.vstack((actions, new_entries))
        # actions.resize((curr_shape[0] + 10, curr_shape[1]))
        # actions[curr_shape[0]:] = new_entries
        out.create_dataset("data/{}/actions".format(ep), data=new_actions)

        states = f["data/{}/states".format(ep)][()]
        out.create_dataset("data/{}/states".format(ep), data=states)
        out['data'].attrs['env_args']  = f['data'].attrs['env_args']


out.close()