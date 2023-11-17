import argparse
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode-data-path", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.mode_data_path, "rb") as f:
        mode_data = pickle.load(f)

    gt_fl = []
    pred_fl = []
    for mode_data_i in mode_data:
        pred_len = mode_data_i["prediction"].shape[0]
        gt_len = mode_data_i["gt"].shape[0]
        assert pred_len == gt_len, \
            f"One of the trajectories has inconsistent length between prediction ({pred_len}) and ground truth ({gt_len})"
        gt_fl.append(mode_data_i["gt"])
        pred_fl.append(mode_data_i["prediction"])
    gt_fl = np.concatenate(gt_fl)
    pred_fl = np.concatenate(pred_fl)
    
    n_trajs = len(mode_data)
    n_steps = gt_fl.shape[0]
    
    last_mode_idx = gt_fl.max()
    second_last_mode_idx = last_mode_idx - 1
    gt_fl[gt_fl == last_mode_idx] = second_last_mode_idx # HACK: merge the last mode the the second last
    
    accuracy = accuracy_score(gt_fl, pred_fl)
    c_mat_norm_method = ["true", "pred", "all"][0]
    c_mat = confusion_matrix(gt_fl, pred_fl, normalize=c_mat_norm_method)
    pr_average_method = ["micro", "macro", "weighted", "samples"][2]
    precision = precision_score(gt_fl, pred_fl, average=pr_average_method)
    recall = recall_score(gt_fl, pred_fl, average=pr_average_method)
    
    print(f"Number of trajectories: {n_trajs}")
    print(f"Number of total steps: {n_steps}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision (average by {pr_average_method}): {precision}")
    print(f"Recall (average by {pr_average_method}): {recall}")
    print(f"Confusion Matrix (normalize along {c_mat_norm_method}):\n{c_mat}")


if __name__ == "__main__":
    main()
