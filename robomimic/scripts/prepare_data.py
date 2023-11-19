import h5py
import os 
import numpy as np
import robosuite.utils.transform_utils as T


# def prepare_data_can(datapath, filename, max_len=200):
#     f = h5py.File(os.path.join(datapath, filename + '.hdf5'), 'r')
#     succ_states = []
#     fail_states = []
#     for demo in f['data'].keys():
#         eef_pos = f['data'][demo]['robot0_eef_pos'][:]
#         gripper = f['data'][demo]['robot0_gripper_qpos'][:]
#         gripper_state = ((gripper[:, 0] - gripper[:, 1]) > 0.06).astype(np.float32).reshape(-1, 1)
#         can_pos = f['data'][demo]['Can_pos'][:]
#         state = np.concatenate((can_pos-eef_pos, gripper, can_pos), axis=1)
#         assert len(state) < max_len
#         state = np.pad(state, ((0, max_len - len(state)), (0, 0)), 'edge')
#         if 'succ' in demo:
#             succ_states.append(state)
#         else:
#             fail_states.append(state)
#     return np.stack(succ_states), np.stack(fail_states)

def prepare_data_can(datapath, filename, max_len=200):
    f = h5py.File(os.path.join(datapath, filename + '.hdf5'), 'r')
    succ_states = []
    fail_states = []
    et_succ_states = []
    et_fail_states = []
    for demo in f['data'].keys():      
        # eef_pos = f['data'][demo]['robot0_eef_pos'][:]
        # gripper = f['data'][demo]['robot0_gripper_qpos'][:]
        # gripper_state = ((gripper[:, 0] - gripper[:, 1]) > 0.06).astype(np.float32).reshape(-1, 1)
        # obj_pos = f['data'][demo]['Can_pos'][:]
        # state = np.concatenate((obj_pos-eef_pos, gripper, obj_pos), axis=1)
        state = []
        for site in f['data'][demo].keys():
            if 'success' in site:
                continue
            if 'states' in site:
                continue
            state.append(f['data'][demo][site][:])
        state = np.concatenate(state, axis=1)

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
    return np.stack(succ_states), np.stack(fail_states) #, np.stack(et_succ_states), np.stack(et_fail_states)

def prepare_data_lift(datapath, filename, max_len=200):
    f = h5py.File(os.path.join(datapath, filename + '.hdf5'), 'r')
    succ_states = []
    fail_states = []
    et_succ_states = []
    et_fail_states = []
    for demo in f['data'].keys():
        # eef_pos = f['data'][demo]['robot0_eef_pos'][:]
        # gripper = f['data'][demo]['robot0_gripper_qpos'][:]
        # gripper_state = ((gripper[:, 0] - gripper[:, 1]) > 0.06).astype(np.float32).reshape(-1, 1)
        # obj_pos = f['data'][demo]['cube_pos'][:]
        # state = np.concatenate((obj_pos-eef_pos, gripper, obj_pos), axis=1)
        state = []
        for site in f['data'][demo].keys():
            if 'success' in site:
                continue
            if 'states' in site:
                continue
            state.append(f['data'][demo][site][:])
        state = np.concatenate(state, axis=1)

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

    num_of_demo_to_discard = 0
    for demo in f['data'].keys():
        # gripper = f['data'][demo]['robot0_gripper_qpos'][:]
        # print(demo)
        # state = []
        # for target in ['SquareNut_handle_site','SquareNut_center_site','SquareNut_side_site']:
        #     # state.append(f['data'][demo][target][:])
        #     state.append(f['data'][demo]['peg_site'][:] - f['data'][demo][target][:])
        #     for keypoint in ['gripper0_grip_site', 'gripper0_left_ee_site', 'gripper0_right_ee_site']:
        #         state.append(f['data'][demo][target][:] - f['data'][demo][keypoint][:])
        # state.append(gripper)
        # state = np.concatenate(state, axis=1)

        state = []
        for site in f['data'][demo].keys():
            if 'success' in site:
                continue
            if 'states' in site:
                continue
            state.append(f['data'][demo][site][:])
        # state.append(gripper)
        state = np.concatenate(state, axis=1)

        try:
            assert len(state) < max_len
        except:
            num_of_demo_to_discard += 1
            print("exceeding max_len of: ", max_len, ", traj has length: ", len(state), ", discarded: ", num_of_demo_to_discard)
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


    np.save(os.path.join(args.datapath, args.filename + 'full_feature_succ.npy'), succ_states[:args.n])
    np.save(os.path.join(args.datapath, args.filename + 'full_feature_fail.npy'), fail_states[:args.n])

    # np.save(os.path.join(args.datapath, args.filename + 'full_feature_et_succ.npy'), et_succ_states[:args.n])
    # np.save(os.path.join(args.datapath, args.filename + 'full_feature_et_fail.npy'), et_fail_states[:args.n])