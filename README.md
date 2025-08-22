This is the source code and the camera ready version (in PDF) for our IJCNN 2023 paper “Deep Reinforcement Learning based Multi-task Automated Channel Pruning for DNNs”.

> Model compression is a key technique that enables deploying Deep Neural Networks (DNNs) on Internet-of-Things (IoT) devices with constrained computing resources and limited power budgets. Channel pruning has become one of the representative compression approaches, but how to determine the compression ratio for different layers of a model still remains as a challenging task. Current automated pruning solutions address this issue by searching for an optimal strategy according to the target compression ratio. Nevertheless, when given a series of tasks with multiple compression ratios and different training datasets, these approaches have to carry out the pruning process repeatedly, which is inefficient and time-consuming. In this paper, we propose a Multi-Task Automated Channel Pruning (MTACP) framework, which can simultaneously generate a number of feasible compressed models satisfying different task demands for a target DNN model. To learn MTACP, the layer-by-layer multi-task channel pruning process is transformed into a Markov Decision Process (MDP), which seeks to solve a series of decisionmaking problems. Based on this MDP, we propose an actorcritic-based multi-task Reinforcement Learning (RL) algorithm to learn the optimal policy, working based on the IMPortance weighted Actor-Learner Architectures (IMPALA). IMPALA is known as a distributed RL architecture, in which the learner can learn from a set of actors that continuously generate trajectories of experience in their own environments. Extensive experiments on CIFAR10/100 and FLOWER102 datasets for MTACP demonstrate its unique capability for multi-task settings, as well as its superior performance over state-of-the-art solutions.

> 模型压缩是在计算资源受限、功耗预算有限的物联网（IoT）设备上部署深度神经网络（DNN）的关键技术。通道剪枝已成为代表性压缩方法之一，但如何确定模型中不同层的压缩比例仍是一个具有挑战性的任务。当前自动化剪枝方案通过根据目标压缩比搜索最优策略来解决这一问题。然而，当面对具有多个压缩比例和不同训练数据集的系列任务时，这些方法需要重复执行剪枝过程，效率低下且耗时。本文提出多任务自动通道剪枝（MTACP）框架，能够为目标DNN模型同时生成多个满足不同任务需求的可行压缩模型。为学习MTACP，我们将逐层多任务通道剪枝过程转化为马尔可夫决策过程（MDP），旨在解决一系列决策问题。基于此MDP，我们提出一种基于行动者-评论家架构的多任务强化学习（RL）算法，依托重要性加权行动者-学习者架构（IMPALA）来学习最优策略。IMPALA是一种分布式强化学习架构，其学习者可从一组在各自环境中持续生成经验轨迹的行动者中进行学习。在CIFAR10/100和FLOWER102数据集上进行的大量实验表明，MTACP不仅具有多任务场景下的独特能力，其性能也优于最先进的解决方案。

You can find the paper from this repo [link](https://github.com/fangvv/MTACP/blob/main/MTACP%20Paper%20Camera%20Ready.pdf) or [ieeexplore](https://doi.org/10.1109/IJCNN54540.2023.10191092).

## Citation

    @inproceedings{ma2023deep,
        title={Deep Reinforcement Learning Based Multi-Task Automated Channel Pruning for DNNs},
        author={Ma, Xiaodong and Fang, Weiwei},
        booktitle={2023 International Joint Conference on Neural Networks (IJCNN)},
        pages={1--9},
        year={2023},
        organization={IEEE}
    }

# autoPrune
automatic model channel pruning using distributed reinforcement learning
```sh
git clone https://gitee.com/rjgcmxd/multi-task-automated-channel-pruning-mtacp.git
```
```sh
cd multi-task-automated-channel-pruning-mtacp
```
#### 用 conda_requirement 创建新环境
```sh
conda create --name <your env name> --file conda_requirements.txt
```
```sh
conda activate <your env name>
```

#### 在新环境中安装 packages
```sh
pip install -r pip_requirement.txt
```
