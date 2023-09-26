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
        
    n_trajs = len(mode_data)
    gt_fl = np.concatenate([v["gt"] for v in mode_data])
    last_mode_idx = gt_fl.max()
    second_last_mode_idx = last_mode_idx - 1
    gt_fl[gt_fl == last_mode_idx] = second_last_mode_idx # HACK: merge the last mode the the second last
    pred_fl = np.concatenate([v["prediction"] for v in mode_data])
    n_steps = min(gt_fl.shape[0], pred_fl.shape[0])
    gt_fl = gt_fl[:n_steps]
    pred_fl = pred_fl[:n_steps]
    
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
