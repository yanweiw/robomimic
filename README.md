# robomimic

To run inference on trained model. `pert_01_1000` indicates perturbation magnitude is 0.1m and this dataset contains 1000 runs off the original demos (200 of them in total). `--n` shows the index of the trajectory to run inference on. Each inference run will run about 40 trajs. `--run_path` indicates the run path shown on the wandb website. For example, the following inference uses model trained from experiment `clean-totem-53` and on the website they show the corresponding run_path under overview tab. You can vary the --epoch number frmo 0-199k. Usually 99k is a good point and 199k typically overfits. You can also vary the guess_idx from 0 to 9. 
```
python infer.py --dataset ../../shared_runs/can/pert_01_1000.hdf5 --render_image_names agentview --video_path ../../shared_runs/can/pert_01_1000_16.mp4 --n 0 --run_path mode_learning/robosuite/ff5lyb3e --epoch 99000 --guess_idx 0 --weight_dir ../../shared_runs/weights
```


To visualize demo dataset
```
python playback_dataset.py 
--dataset ../../datasets/square/ph/low_dim_v141.hdf5 
--render_image_names agentview frontview 
--video_path /tmp/playback_dataset.mp4 
--n 20
```

To playback dataset
```
python playback_dataset.py --dataset ../../datasets/can/ph/low_dim_v141.hdf5 --render_iamge_names agentview --render (or --video_path) (--use_actions) --n 10 --ic 100
```

<p align="center">
  <img width="24.0%" src="docs/images/task_lift.gif">
  <img width="24.0%" src="docs/images/task_can.gif">
  <img width="24.0%" src="docs/images/task_tool_hang.gif">
  <img width="24.0%" src="docs/images/task_square.gif">
  <img width="24.0%" src="docs/images/task_lift_real.gif">
  <img width="24.0%" src="docs/images/task_can_real.gif">
  <img width="24.0%" src="docs/images/task_tool_hang_real.gif">
  <img width="24.0%" src="docs/images/task_transport.gif">
 </p>

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

## Mode Classiciation Evaluation
* Examples
  ```
  $ python infer.py --dataset ../../shared_runs/can/pert_01_1000.hdf5 
                    --n 0 
                    --run_path mode_learning/robosuite/ff5lyb3e 
                    --epoch 99000 
                    --guess_idx 0 
                    --weight_dir ../../shared_runs/weights 
                    --render_image_names agentview 
                    --video_path ../../shared_runs/can/pert_01_1000_16.mp4 
                    --mode_data_path ../../shared_runs/can/pert_01_1000_16.pkl
  $ python scripts/plot_mode_comparison.py --mode-data-path ../shared_runs/can/pert_01_1000_16.pkl
  ```

-------
## Latest Updates
- [07/03/2023] **v0.3.0**: BC-Transformer and IQL :brain:, support for DeepMind MuJoCo bindings :robot:, pre-trained image reps :eye:, wandb logging :chart_with_upwards_trend:, and more
- [05/23/2022] **v0.2.1**: Updated website and documentation to feature more tutorials :notebook_with_decorative_cover:
- [12/16/2021] **v0.2.0**: Modular observation modalities and encoders :wrench:, support for [MOMART](https://sites.google.com/view/il-for-mm/home) datasets :open_file_folder: [[release notes]](https://github.com/ARISE-Initiative/robomimic/releases/tag/v0.2.0) [[documentation]](https://robomimic.github.io/docs/v0.2/introduction/overview.html)
- [08/09/2021] **v0.1.0**: Initial code and paper release

-------

## Colab quickstart
Get started with a quick colab notebook demo of robomimic without installing anything locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b62r_km9pP40fKF0cBdpdTO2P_2eIbC6?usp=sharing)


-------

**robomimic** is a framework for robot learning from demonstration.
It offers a broad set of demonstration datasets collected on robot manipulation domains and offline learning algorithms to learn from these datasets.
**robomimic** aims to make robot learning broadly *accessible* and *reproducible*, allowing researchers and practitioners to benchmark tasks and algorithms fairly and to develop the next generation of robot learning algorithms.

## Core Features

<p align="center">
  <img width="50.0%" src="docs/images/core_features.png">
 </p>

<!-- **Standardized Datasets**
- Simulated and real-world tasks
- Multiple environments and robots
- Diverse human-collected and machine-generated datasets

**Suite of Learning Algorithms**
- Imitation Learning algorithms (BC, BC-RNN, HBC)
- Offline RL algorithms (BCQ, CQL, IRIS, TD3-BC)

**Modular Design**
- Low-dim + Visuomotor policies
- Diverse network architectures
- Support for external datasets

**Flexible Workflow**
- Hyperparameter sweep tools
- Dataset visualization tools
- Generating new datasets -->


## Reproducing benchmarks

The robomimic framework also makes reproducing the results from different benchmarks and datasets easy. See the [datasets page](https://robomimic.github.io/docs/datasets/overview.html) for more information on downloading datasets and reproducing experiments.

## Troubleshooting

Please see the [troubleshooting](https://robomimic.github.io/docs/miscellaneous/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/ARISE-Initiative/robomimic/issues) on our github page.

## Contributing to robomimic
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.
The project originally began development in late 2018 by researchers in the [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL).
Now it is actively maintained and used for robotics research projects across multiple labs.
We welcome community contributions to this project.
For details please check our [contributing guidelines](https://robomimic.github.io/docs/miscellaneous/contributing.html).

## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```
