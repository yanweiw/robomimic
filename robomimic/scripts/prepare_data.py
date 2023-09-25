import h5py
import os 
import numpy as np
import robosuite.utils.transform_utils as T


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

def prepare_data_square(datapath, filename, max_len=200):
    f = h5py.File(os.path.join(datapath, filename + '.hdf5'), 'r')
    succ_states = []
    fail_states = []
    et_succ_states = []
    et_fail_states = []
    for demo in f['data'].keys():
        nut_pos = f['data'][demo]['SquareNut_pos'][:]
        nut_quat = f['data'][demo]['SquareNut_quat'][:]
        nut2eef_pos = f['data'][demo]['SquareNut_to_robot0_eef_pos'][:]
        nut2eef_quat = f['data'][demo]['SquareNut_to_robot0_eef_quat'][:]

        # nut_euler = []
        # for i in range(len(nut_quat)):
        #     nut_euler.append(T.quat2axisangle(nut_quat[i]))
        # nut_euler = np.array(nut_euler)
        # nut2eef_euler = []
        # for i in range(len(nut2eef_quat)):
        #     nut2eef_euler.append(T.quat2axisangle(nut2eef_quat[i]))
        # nut2eef_euler = np.array(nut2eef_euler)

        # from IPython import embed; embed()
        eef_pos = f['data'][demo]['robot0_eef_pos'][:]
        gripper = f['data'][demo]['robot0_gripper_qpos'][:]
        gripper_state = ((gripper[:, 0] - gripper[:, 1]) > 0.06).astype(np.float32).reshape(-1, 1)
        # obj_pos = f['data'][demo]['cube_pos'][:]
        state = np.concatenate((nut_pos, nut_quat, nut2eef_pos, nut2eef_quat, gripper), axis=1)
        try:
            assert len(state) < max_len
        except:
            print("exceeding max_len of: ", max_len, ", traj has length: ", len(state))
            continue
            
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
    if args.task == 'square':
        succ_states, fail_states, et_succ_states, et_fail_states = prepare_data_square(args.datapath, args.filename)

# for i, j in enumerate([0, 1, 2]):
#     ...:     plt.plot(fail[:10, :, j].T, c='r', alpha=0.2*i + 0.2)
#     ...:     plt.plot(succ[:10, :, j].T, c='g', alpha=0.2*i + 0.2)


    np.save(os.path.join(args.datapath, args.filename + '_succ.npy'), succ_states[:args.n])
    np.save(os.path.join(args.datapath, args.filename + '_fail.npy'), fail_states[:args.n])
    np.save(os.path.join(args.datapath, args.filename + '_et_succ.npy'), et_succ_states[:args.n])
    np.save(os.path.join(args.datapath, args.filename + '_et_fail.npy'), et_fail_states[:args.n])