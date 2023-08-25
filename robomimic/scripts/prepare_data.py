import h5py
import os 
import numpy as np

def prepare_data_can(datapath, filename, max_len=200):
    f = h5py.File(os.path.join(datapath, filename), 'r')
    states = []
    for demo in f['data'].keys():
        eef_pos = f['data'][demo]['robot0_eef_pos'][:]
        gripper = f['data'][demo]['robot0_gripper_qpos'][:]
        # eef_pos = np.pad(eef_pos, ((0, max_len - len(eef_pos)), (0, 0)), 'edge')
        # eef_traj.append(eef_pos)
        can_pos = f['data'][demo]['Can_pos'][:]
        state = np.concatenate((eef_pos, gripper, can_pos), axis=1)
        assert len(state) < max_len
        state = np.pad(state, ((0, max_len - len(state)), (0, 0)), 'edge')
        states.append(state)
    return np.stack(states)
