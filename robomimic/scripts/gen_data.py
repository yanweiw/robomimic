import os
import h5py
import argparse
import numpy as np

def gather_demonstrations_as_hdf5(directory, out_dir, env_info=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_succ_path = os.path.join(out_dir, "succ_" + args.save_name +".hdf5")
    hdf5_fail_path = os.path.join(out_dir, "fail_" + args.save_name +".hdf5")
    fs = h5py.File(hdf5_succ_path, "w")
    ff = h5py.File(hdf5_fail_path, "w")

    # store some metadata in the attributes of one group
    grp_s = fs.create_group("data")
    grp_f = ff.create_group("data")

    num_succ_eps = 0
    num_fail_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        data_path = os.path.join(directory, ep_directory, "obs.npz")
        data = np.load(data_path, allow_pickle=True)
        # from IPython import embed; embed()
        if data['success']:
            tag = str(num_succ_eps).zfill(6)
            ep_succ_grp = grp_s.create_group("demo_{}".format(tag))
            for key in data.files:
                assert key != "env_args"
                ep_succ_grp.create_dataset(key, data=data[key])
                 
            print("Number {}th successful demo has been saved".format(tag))
            num_succ_eps += 1
        else:
            tag = str(num_fail_eps).zfill(6)
            ep_fail_grp = grp_f.create_group("demo_{}".format(tag))
            for key in data.files:
                assert key != "env_args"
                ep_fail_grp.create_dataset(key, data=data[key])

            print("Number {}th failing trajectory has been saved".format(tag))
            num_fail_eps += 1

    with open(os.path.join(directory, ep_directory, 'env_args.txt'), "r") as f:
        env_args = f.read()
    grp_s.attrs['env_args'] = env_args       

    with open(os.path.join(directory, ep_directory, 'env_args.txt'), "r") as f:
        env_args = f.read()
    grp_f.attrs['env_args'] = env_args                     
            
        # states = []
        # actions = []
        # success = False

    #     for state_file in sorted(glob(state_paths)):
    #         dic = np.load(state_file, allow_pickle=True)
    #         env_name = str(dic["env"])

    #         states.extend(dic["states"])
    #         for ai in dic["action_infos"]:
    #             actions.append(ai["actions"])
    #         success = success or dic["successful"]

    #     if len(states) == 0:
    #         continue

    #     # Add only the successful demonstration to dataset
    #     if success:
    #         print("Demonstration is successful and has been saved")
    #         # Delete the last state. This is because when the DataCollector wrapper
    #         # recorded the states and actions, the states were recorded AFTER playing that action,
    #         # so we end up with an extra state at the end.
    #         del states[-1]
    #         assert len(states) == len(actions)

    #         num_eps += 1
    #         ep_data_grp = grp.create_group("demo_{}".format(num_eps))

    #         # store model xml as an attribute
    #         xml_path = os.path.join(directory, ep_directory, "model.xml")
    #         with open(xml_path, "r") as f:
    #             xml_str = f.read()
    #         ep_data_grp.attrs["model_file"] = xml_str

    #         # write datasets for states and actions
    #         ep_data_grp.create_dataset("states", data=np.array(states))
    #         ep_data_grp.create_dataset("actions", data=np.array(actions))
    #     else:
    #         print("Demonstration is unsuccessful and has NOT been saved")

    # # write dataset attributes (metadata)
    # now = datetime.datetime.now()
    # grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    # grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    # grp.attrs["repository_version"] = suite.__version__
    # grp.attrs["env"] = env_name
    # grp.attrs["env_info"] = env_info

    fs.close()
    ff.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        help="path to directory containing perturbed demonstrations",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="path to hdf5 dataset to be saved",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        help="name of the saved hdf5 dataset",
    )
        
    args = parser.parse_args()
    gather_demonstrations_as_hdf5(args.in_dir, args.out_dir)