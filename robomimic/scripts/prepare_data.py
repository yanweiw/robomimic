import h5py
import os 
import numpy as np

def prepare_data_can(datapath, filename, max_len=200):
    f = h5py.File(os.path.join(datapath, filename + '.hdf5'), 'r')
    succ_states = []
    fail_states = []
    for demo in f['data'].keys():
        eef_pos = f['data'][demo]['robot0_eef_pos'][:]
        gripper = f['data'][demo]['robot0_gripper_qpos'][:]
        # eef_pos = np.pad(eef_pos, ((0, max_len - len(eef_pos)), (0, 0)), 'edge')
        # eef_traj.append(eef_pos)
        can_pos = f['data'][demo]['Can_pos'][:]
        state = np.concatenate((eef_pos, gripper, can_pos), axis=1)
        assert len(state) < max_len
        state = np.pad(state, ((0, max_len - len(state)), (0, 0)), 'edge')
        if 'succ' in demo:
            succ_states.append(state)
        else:
            fail_states.append(state)
    return np.stack(succ_states), np.stack(fail_states)


if __name__  == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        help="path to directory containing perturbed demonstrations",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="name of the saved hdf5 dataset",
    )
    # parser.add_argument(
    #     "--savename",
    #     type=str,
    #     help="path to hdf5 dataset to be saved",
    # )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="number of trajectories",
    )
    args = parser.parse_args()

    succ_states, fail_states = prepare_data_can(args.datapath, args.filename)
    np.save(os.path.join(args.datapath, args.filename + '_succ.npy'), succ_states[:args.n])
    np.save(os.path.join(args.datapath, args.filename + '_fail.npy'), fail_states[:args.n])