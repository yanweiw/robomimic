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
        gripper_state = ((gripper[:, 0] - gripper[:, 1]) > 0.06).astype(np.float32).reshape(-1, 1)
        can_pos = f['data'][demo]['Can_pos'][:]
        state = np.concatenate((can_pos-eef_pos, gripper, can_pos), axis=1)
        assert len(state) < max_len
        state = np.pad(state, ((0, max_len - len(state)), (0, 0)), 'edge')
        if 'succ' in demo:
            succ_states.append(state)
        else:
            fail_states.append(state)
    return np.stack(succ_states), np.stack(fail_states)

def prepare_data_lift(datapath, filename, max_len=200):
    f = h5py.File(os.path.join(datapath, filename + '.hdf5'), 'r')
    succ_states = []
    fail_states = []
    et_succ_states = []
    et_fail_states = []
    for demo in f['data'].keys():
        eef_pos = f['data'][demo]['robot0_eef_pos'][:]
        gripper = f['data'][demo]['robot0_gripper_qpos'][:]
        gripper_state = ((gripper[:, 0] - gripper[:, 1]) > 0.06).astype(np.float32).reshape(-1, 1)
        obj_pos = f['data'][demo]['cube_pos'][:]
        state = np.concatenate((obj_pos-eef_pos, gripper, obj_pos), axis=1)
        assert len(state) < max_len
        state = np.pad(state, ((0, max_len - len(state)), (0, 0)), 'edge')
        if 'succ' in demo:
            if 'et' in demo:
                et_succ_states.append(state)
            else:
                succ_states.append(state)
        else:
            if 'et' in demo:
                et_fail_states.append(state)
            else:
                fail_states.append(state)
    return np.stack(succ_states), np.stack(fail_states), np.stack(et_succ_states), np.stack(et_fail_states)

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

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="name of the task",
    )

    args = parser.parse_args()
    if args.task == 'can':
        succ_states, fail_states = prepare_data_can(args.datapath, args.filename)
    if args.task == 'lift':
        succ_states, fail_states, et_succ_states, et_fail_states = prepare_data_lift(args.datapath, args.filename)
    np.save(os.path.join(args.datapath, args.filename + '_succ.npy'), succ_states[:args.n])
    np.save(os.path.join(args.datapath, args.filename + '_fail.npy'), fail_states[:args.n])
    np.save(os.path.join(args.datapath, args.filename + '_et_succ.npy'), et_succ_states[:args.n])
    np.save(os.path.join(args.datapath, args.filename + '_et_fail.npy'), et_fail_states[:args.n])