# MTACP

This is the source code for our paper: **Deep Reinforcement Learning based Multi-task Automated Channel Pruning for DNNs**. A brief introduction of this work is as follows:

Model compression is a key technique that enables deploying Deep Neural Networks (DNNs) on Internet-of-Things (IoT) devices with constrained computing resources and limited power budgets. Channel pruning has become one of the representative compression approaches, but how to determine the compression ratio for different layers of a model still remains as a challenging task. Current automated pruning solutions address this issue by searching for an optimal strategy according to the target compression ratio. Nevertheless, when given a series of tasks with multiple compression ratios and different training datasets, these approaches have to carry out the pruning process repeatedly, which is inefficient and time-consuming. In this paper, we propose a Multi-Task Automated Channel Pruning (MTACP) framework, which can simultaneously generate a number of feasible compressed models satisfying different task demands for a target DNN model. To learn MTACP, the layer-by-layer multi-task channel pruning process is transformed into a Markov Decision Process (MDP), which seeks to solve a series of decision-making problems. Based on this MDP, we propose an actor-critic-based multi-task Reinforcement Learning (RL) algorithm to learn the optimal policy, working based on the IMPortance weighted Actor-Learner Architectures (IMPALA). IMPALA is known as a distributed RL architecture, in which the learner can learn from a set of actors that continuously generate trajectories of experience in their own environments. Extensive experiments on CIFAR10/100 and FLOWER102 datasets for MTACP demonstrate its unique capability for multi-task settings, as well as its superior performance over state-of-the-art solutions.

模型压缩是在计算资源受限、功耗预算有限的物联网（IoT）设备上部署深度神经网络（DNN）的关键技术。通道剪枝已成为代表性压缩方法之一，但如何确定模型中不同层的压缩比例仍是一个具有挑战性的任务。当前自动化剪枝方案通过根据目标压缩比搜索最优策略来解决这一问题。然而，当面对具有多个压缩比例和不同训练数据集的系列任务时，这些方法需要重复执行剪枝过程，效率低下且耗时。本文提出多任务自动通道剪枝（MTACP）框架，能够为目标DNN模型同时生成多个满足不同任务需求的可行压缩模型。为学习MTACP，我们将逐层多任务通道剪枝过程转化为马尔可夫决策过程（MDP），旨在解决一系列决策问题。基于此MDP，我们提出一种基于行动者-评论家架构的多任务强化学习（RL）算法，依托重要性加权行动者-学习者架构（IMPALA）来学习最优策略。IMPALA是一种分布式强化学习架构，其学习者可从一组在各自环境中持续生成经验轨迹的行动者中进行学习。在CIFAR10/100和FLOWER102数据集上进行的大量实验表明，MTACP不仅具有多任务场景下的独特能力，其性能也优于最先进的解决方案。

This work will be published by IJCNN 2023. Click [here](https://doi.org/10.1109/IJCNN54540.2023.10191092) for our paper online, or you can find the camera ready PDF in this repo [link](https://github.com/fangvv/MTACP/blob/main/MTACP%20Paper%20Camera%20Ready.pdf).

## Required software

- Python 3.7+

- PyTorch 1.9

- Torchvision 0.2.1

- NumPy

- scikit-learn

- ptflops

- TensorboardX

## Project Structure

```
MTACP/
├── config/                        # JSON configs for each (model, dataset) task
│   ├── auto_prune_impala_cifar10.json
│   ├── auto_prune_impala_cifar100*.json
│   ├── auto_prune_impala_flower*.json
│   └── ...
├── core/                          # IMPALA core: V-trace, env wrapper, profiling
│   ├── environment.py             # Gym-style env wrapper for the pruning environment
│   ├── file_writer.py
│   ├── prof.py                    # Profiling utilities
│   └── vtrace.py                  # V-trace off-policy correction
├── env/                           # Channel pruning environments & reward functions
│   ├── channel_pruning_env_mobilenet.py  # Pruning env for MobileNet
│   ├── channel_pruning_env_vgg16.py      # Pruning env for VGG-16
│   ├── rewards.py                 # Reward: acc_reward, acc_flops_reward, etc.
│   └── rewards_mxd.py
├── lib/                           # Shared libs: DDPG agent, data loaders, utils
│   ├── agent.py                   # Actor / Critic networks, DDPG agent
│   ├── data.py                    # Dataset loaders & splits
│   ├── memory.py                  # Experience replay buffer
│   ├── net_measure.py             # FLOPs / params measurement
│   ├── thop/                      # FLOPs counter (vendored)
│   └── utils.py
├── models/                        # DNN architectures used in the experiments
│   ├── mobilenet.py
│   ├── mobilenet_v2.py
│   ├── resnet.py
│   └── vgg_cifar.py
├── scripts/                       # Ready-to-run shell scripts
│   ├── search_mobilenet_0.5flops.sh     # Run multi-task search
│   ├── finetune_mobilenet_0.5flops.sh   # Fine-tune the searched model
│   └── export_mobilenet_0.5flops.sh     # Export the pruned model
├── logs/                          # Training logs, checkpoints, exported models
├── amc_search.py                  # Entry: AMC baseline search (DDPG-based)
├── amc_fine_tune.py               # Entry: Fine-tune the searched pruned model
├── cacp_search.py                 # Entry: CACP baseline search
├── eval_mobilenet.py              # Entry: Evaluate a pruned MobileNet
├── impala_auto_pruning.py         # Entry: Main MTACP algorithm (IMPALA-based multi-task pruning)
├── atari_wrappers.py              # Atari wrappers reused by the IMPALA pipeline
├── utils.py                       # Top-level helpers (model loading, config, etc.)
├── conda_requirements.txt
├── pip_requirement.txt
└── README.md
```

## Core Modules

### Pruning Environment (`env/channel_pruning_env_*.py`)

The environment class that models the layer-by-layer channel pruning process as an MDP. Key attributes:

| Attribute | Description |
|---|---|
| `lbound` / `rbound` | Lower / upper bound of the per-layer preserve ratio (e.g. `0.2` / `1.0`) |
| `preserve_ratio` | Target overall FLOPs/params compression ratio for the task |
| `channel_round` | Round the number of kept channels to a multiple of this value (e.g. `8`) |
| `n_calibration_batches` | Batches used to estimate channel importance |
| `n_points_per_layer` | Number of importance samples per layer |
| `prunable_layer_types` | Layer types eligible for pruning: `Conv2d`, `Linear` |
| `acc_metric` | Accuracy metric: `acc1` or `acc5` |
| `use_real_val` | If true, use the real validation set to compute accuracy |

**State space** (continuous): the state encodes the current DNN to be pruned, including the preserve ratio of the next layer, the FLOPs/params of the next prunable layer, the input/output channel counts of that layer, and cumulative statistics over previous layers.

**Action space** (continuous, in `[lbound, rbound]`): the preserve ratio of the next prunable layer. The agent decides how aggressively to prune the current layer.

**Key methods:**

- `reset()` — Restore the model from the backup, reset the layer pointer, and return the initial state.

- `step(action)` — Apply the chosen preserve ratio to the current layer, advance the layer pointer, and return `(next_state, reward, done, info)`. When all prunable layers are processed, the environment validates the model, computes the reward, and resets.

- `_validate()` — Evaluate the current pruned model on the validation set and return top-1 (or top-5) accuracy.

- `_get_reward()` — Call into `env.rewards` to convert `(acc, flops)` into a scalar reward.

**Reward:** configurable in `env/rewards.py`:

- `acc_reward` — `acc * 0.01` (maximize accuracy).
- `acc_flops_reward` — `-((100 - acc) / 100) * log(flops)` (trade-off accuracy and FLOPs).
- `acc_reward_mxd` — Centered accuracy reward used in the multi-task setting.

### IMPALA Learner (`impala_auto_pruning.py`)

The main MTACP algorithm. Multiple actors run in parallel, each holding a different `(model, dataset, target_ratio)` task. They generate trajectories of pruning actions; a central learner updates the policy using V-trace off-policy correction.

**Network architecture (Actor / Critic):**

| Network | Layers | Output Activation |
|---|---|---|
| Actor | `state → 400(relu) → 300(relu) → action` | `sigmoid` (action in `[0, 1]`, mapped to `[lbound, rbound]`) |
| Critic | `state(400) + action(400) → 300(relu) → 1` | linear (value) |

The Actor outputs a continuous Gaussian policy (mean & std) per layer, sampled to produce the preserve ratio.

**Key hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `num_actors` | `1` (default) | Number of parallel actor processes |
| `total_steps` | `10000` (default) | Total environment steps for training |
| `batch_size` | `32` | Training batch size per env |
| `discount` | `0.99` | Discount factor γ |
| `entropy_cost` | `0.01` | Entropy bonus for exploration |
| `baseline_cost` | `0.5` | Critic loss weight |
| `learning_rate` | `1e-3` | RMSProp learning rate |
| `clip_rho_threshold` | `1.0` | V-trace importance-weight clip |
| `clip_pg_rho_threshold` | `1.0` | V-trace policy-gradient clip |

**Key methods:**

- `train()` — Spawn actors and learner, batch trajectories, apply V-trace, and update the network with RMSProp.

- `act()` — Sample a continuous action from the current policy and execute `env.step()`.

- `compute_vtrace()` — Apply V-trace to off-policy trajectories to obtain corrected advantages and value targets.

- `file_writer` — Stream TensorBoard logs (loss, reward, mean episode return) to `logs/`.

### V-Trace (`core/vtrace.py`)

Implements the V-trace off-policy correction used by IMPALA. It computes:

- `vs` — the value targets for the value function,
- `pg_advantages` — the policy-gradient advantages,
- `log_rhos` — the clipped importance sampling log-ratios.

Two variants are provided: `from_logits` (continuous Gaussian policy) and `from_importance_weights` (discrete policy).

### Multi-Task Configuration (`config/`)

Each JSON file describes a single `(model, dataset, target_ratio)` task. Sample `auto_prune_impala_cifar10.json`:

```json
{
    "dataset": "cifar10",
    "preserve_ratio": 0.5,
    "model": "mobilenet",
    "n_data_worker": 8,
    "lbound": 0.2,
    "rbound": 1,
    "reward": "acc_reward",
    "data_root": "/../dataset/cifar10",
    "ckpt_path": "./checkpoints/mobilenetamc_cifar10.pth",
    "seed": 2022,
    "batch_size": 32,
    "channel_round": 8,
    "n_points_per_layer": 10,
    "n_calibration_batches": 60
}
```

Multiple tasks (different datasets, ratios) are launched together to form the multi-task MDP.

### Baselines

- **AMC** (`amc_search.py`) — DDPG-based single-task automated channel pruning.
- **CACP** (`cacp_search.py`) — AutoPrune-style continuous-action search.
- **Fine-tune** (`amc_fine_tune.py`) — Standard training of the searched pruned model.
- **Evaluate** (`eval_mobilenet.py`) — Load a pruned checkpoint and report top-1 / top-5 accuracy and FLOPs.

## Usage

```sh
# Clone the repository
git clone https://github.com/fangvv/MTACP.git
cd MTACP

# Create a new conda environment from the lock file
conda create --name mtacp --file conda_requirements.txt
conda activate mtacp

# Install pip dependencies
pip install -r pip_requirement.txt
```

```sh
# Run the main MTACP algorithm (multi-task IMPALA pruning)
python impala_auto_pruning.py --env AutoPrune-v0 --mode train

# Run the AMC baseline (single-task DDPG)
python amc_search.py --job train --model mobilenet --dataset cifar10 \
    --preserve_ratio 0.5 --lbound 0.2 --rbound 1 --reward acc_reward \
    --data_root ./dataset/cifar10/ --ckpt_path ./checkpoints/mobilenetamc_cifar10.pth --seed 2022

# Run the CACP baseline
python cacp_search.py

# Fine-tune the searched pruned model
python -W ignore amc_fine_tune.py --model mobilenet_0.5flops --dataset cifar10 \
    --lr 0.05 --n_gpu 1 --batch_size 256 --n_worker 8 --lr_type cos --n_epoch 150 \
    --data_root ./dataset/cifar10/ --ckpt_path ./checkpoints/mobilenet_cifar10_0.5flops_export.pth.tar

# Evaluate a pruned MobileNet
python eval_mobilenet.py
```

See `scripts/search_mobilenet_0.5flops.sh`, `scripts/finetune_mobilenet_0.5flops.sh`, and `scripts/export_mobilenet_0.5flops.sh` for ready-to-run examples.

## Citation

If you find MTACP useful or relevant to your project and research, please kindly cite our paper:

```
@inproceedings{ma2023deep,
    title={Deep Reinforcement Learning Based Multi-Task Automated Channel Pruning for DNNs},
    author={Ma, Xiaodong and Fang, Weiwei},
    booktitle={2023 International Joint Conference on Neural Networks (IJCNN)},
    pages={1--9},
    year={2023},
    organization={IEEE}
}
```

## For more

We have another work on [UAV-DDPG](https://github.com/fangvv/UAV-DDPG) and [VN-MADDPG](https://github.com/fangvv/VN-MADDPG) for your reference.

## Contact

Xiaodong Ma ([maxdzzu@gmail.com](mailto:maxdzzu@gmail.com))

Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
