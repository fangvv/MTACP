This is the source code and the camera ready version (in PDF) for our IJCNN 2023 paper “Deep Reinforcement Learning based Multi-task Automated Channel Pruning for DNNs”.

> Model compression is a key technique that enables deploying Deep Neural Networks (DNNs) on Internet-of-Things (IoT) devices with constrained computing resources and limited power budgets. Channel pruning has become one of the representative compression approaches, but how to determine the compression ratio for different layers of a model still remains as a challenging task. Current automated pruning solutions address this issue by searching for an optimal strategy according to the target compression ratio. Nevertheless, when given a series of tasks with multiple compression ratios and different training datasets, these approaches have to carry out the pruning process repeatedly, which is inefficient and time-consuming. In this paper, we propose a Multi-Task Automated Channel Pruning (MTACP) framework, which can simultaneously generate a number of feasible compressed models satisfying different task demands for a target DNN model. To learn MTACP, the layer-by-layer multi-task channel pruning process is transformed into a Markov Decision Process (MDP), which seeks to solve a series of decisionmaking problems. Based on this MDP, we propose an actorcritic-based multi-task Reinforcement Learning (RL) algorithm to learn the optimal policy, working based on the IMPortance weighted Actor-Learner Architectures (IMPALA). IMPALA is known as a distributed RL architecture, in which the learner can learn from a set of actors that continuously generate trajectories of experience in their own environments. Extensive experiments on CIFAR10/100 and FLOWER102 datasets for MTACP demonstrate its unique capability for multi-task settings, as well as its superior performance over state-of-the-art solutions.

You can find the paper from this repo [link](https://github.com/fangvv/MTACP/blob/main/MTACP%20Paper%20Camera%20Ready.pdf) or [ieeexplore](https://doi.org/10.1109/IJCNN54540.2023.10191092).

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
