# Code for MXD
# mxdzzu@gmail.com, 20120396@bjtu.edu.cn
import time
import torch
import torch.nn as nn
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
from env.rewards import *
import math
import torchvision.models as models
import numpy as np
import random
import timeit
from random import choice
import os
from copy import deepcopy
from utils import _get_model_and_checkpoint
import copy
import csv
from utils import *
from ptflops import get_model_complexity_info
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class MobileNetAutoPruningEnv:
    def __init__(self, env_index:int, time_stamp):
        prGreen("==========开始初始化ChannelPruningEnv==========")
        self.timer = timeit.default_timer

        cur_path = os.path.abspath(os.path.dirname(__file__))
        self.env_index = env_index

        if env_index % 2 != 0: 
            self.params = read_config(cur_path + "/../config/auto_prune_impala_cifar10.json")
            self.backup_model, self.checkpoint = _get_model_and_checkpoint(self.params.model, self.params.dataset,
                                                            checkpoint_path=self.params.ckpt_path, n_gpu=self.params.n_gpu)
        else: 
            self.params = read_config(cur_path + "/../config/auto_prune_impala_cifar10.json")
            self.backup_model, self.checkpoint = _get_model_and_checkpoint(self.params.model, self.params.dataset,
                                                                checkpoint_path=self.params.ckpt_path,
                                                                n_gpu=self.params.n_gpu)
        self.model = copy.deepcopy(self.backup_model)
        _, self.org_model_size_all = get_model_complexity_info(self.model, (3, 32, 32), as_strings=False, print_per_layer_stat=False, verbose=False)

        # set log
        self.log_path = os.path.expanduser("%s/%s/%s" % (cur_path + "/../logs", time_stamp, env_index))
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(self.log_path + '/log.csv', "a") as f:
            writer = csv.writer(f)
            writer.writerow(["current reward", "acc", "compress ratio", "current policy", "current primes", "episode time(min)", "FLOPs", "params"])
        # set export path
        self.set_export_path(self.log_path+"/prunned_mobilenet.pth")

        # save options
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]
        self.n_data_worker = self.params.n_data_worker
        self.batch_size = self.params.batch_size
        self.data_type = self.params.dataset

        self.lbound = self.params.lbound
        self.rbound = self.params.rbound

        self.preserve_ratio = self._seed(self.params.seed)

        self.use_real_val = self.params.use_real_val

        self.n_calibration_batches = self.params.n_calibration_batches
        self.n_points_per_layer = self.params.n_points_per_layer
        self.channel_round = self.params.channel_round
        self.acc_metric = self.params.acc_metric
        self.data_root = os.path.abspath(os.path.dirname(__file__)) + self.params.data_root

        self.export_model = self.params.export_model
        self.use_new_input = self.params.use_new_input

        self.reduced_params = []

        print("The preserve_ratio of Current Environment: " + str(self.preserve_ratio))
        

        # sanity check
        assert self.preserve_ratio >= self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'

        self._init_data()

        # build subscript(or indexs) of the layer that can be pruned
        self._build_index()
        self.n_prunable_layer = len(self.prunable_idx)

        # extract information for preparing
        self._extract_layer_information()

        # build state (static part)
        self._build_state_embedding()

        self.reset() # restore weight
        # original accuracy before pruning model
        self.org_acc = self._validate(self.val_loader, self.model)
        with open(os.path.expanduser("%s/%s/%s/%s" % (cur_path + "/../logs", time_stamp, env_index, "original.txt")), 'a') as f:
            print('=> original accuracy: {:.3f}%'.format(self.org_acc))
            f.writelines('=> original accuracy: {:.3f}%\n'.format(self.org_acc))
            self.org_model_size = sum(self.wsize_list)
            self.params_bn_size = self.org_model_size_all - self.org_model_size
            print('=> original weight size: {:.4f} M param'.format(self.org_model_size_all * 1. / 1e6)) #  * 1. / 1e6
            f.writelines('=> original weight size: {:.4f} M param \n'.format(self.org_model_size_all * 1. / 1e6)) #  * 1. / 1e6
            self.org_flops = sum(self.flops_list)
            print('=> FLOPs:')
            print([self.layer_info_dict[idx]['flops']/1e6 for idx in sorted(self.layer_info_dict.keys())])
            print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))
            f.writelines('=> original FLOPs: {:.4f} M \n'.format(self.org_flops * 1. / 1e6))
            f.writelines("The preserve_ratio of Current Environment: " + str(self.preserve_ratio))

        self.expected_preserve_computation = self.preserve_ratio * self.org_flops
        self.reward = eval(self.params.reward) # eval("acc_reward")

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None

        self.action_space = (1, )
        self.observation_space = (1, len(self.layer_embedding[0]))

        self.start_time = self.timer()

    def step(self, action):
        if self.visited[self.cur_ind]:
            action = self.strategy_dict[self.prunable_idx[self.cur_ind]][0]
            preserve_idx = self.index_buffer[self.cur_ind]
        else:
            action = self._action_wall(action)  # percentage to preserve
            preserve_idx = None

        # prune and update action
        action, d_prime, preserve_idx = self.prune_kernel(self.prunable_idx[self.cur_ind], action, preserve_idx)

        if not self.visited[self.cur_ind]:
            for group in self.shared_idx:
                if self.cur_ind in group:  # set the shared ones
                    for g_idx in group:
                        self.strategy_dict[self.prunable_idx[g_idx]][0] = action
                        self.strategy_dict[self.prunable_idx[g_idx - 1]][1] = action
                        self.visited[g_idx] = True
                        self.index_buffer[g_idx] = preserve_idx.copy()

        self.strategy.append(action)  # save action to strategy
        self.d_prime_list.append(d_prime) # save d_prime
        self.reduced_params.append(self.wsize_list[self.cur_ind] * (1 - action))

        self.strategy_dict[self.prunable_idx[self.cur_ind]][0] = action
        if self.cur_ind > 0: 
            self.strategy_dict[self.prunable_idx[self.cur_ind - 1]][1] = action

        # all the pruning actions are made
        if self._is_final_layer():
            assert len(self.strategy) == len(self.prunable_idx)
            current_flops = self._cur_flops()
            current_params = self._cur_params()

            acc_t1 = time.time()
            acc = self._validate(self.val_loader, self.model) # inference accuracy
            acc_t2 = time.time() # inference latency
            self.val_time = acc_t2 - acc_t1
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'accuracy': acc, 'strategy': self.strategy.copy()}
            
            reward =  self.reward(self, acc, current_flops)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen('New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}'.format(self.best_reward, acc, compress_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))
                with open(self.log_path + '/log.csv', mode='a') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.best_reward, acc, compress_ratio, self.best_strategy, self.best_d_prime_list, (self.timer() - self.start_time) / 60, current_flops * 1. / 1e6, current_params * 1. / 1e6])
            else:
                with open(self.log_path + '/log.csv', mode='a') as f:
                    writer = csv.writer(f)
                    writer.writerow([reward, acc, compress_ratio, self.strategy.copy(), self.d_prime_list.copy(), (self.timer() - self.start_time) / 60, current_flops * 1. / 1e6, current_params * 1. / 1e6])


            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            if self.export_model:  # export state dict
                torch.save(self.model.state_dict(), self.export_path)
                # return None, None, None, None
            return obs, reward, done, info_set

        info_set = None

        reward = 0

        done = False
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer
        # build next state (in-place modify)
        self.layer_embedding[self.cur_ind][-3] = self._cur_reduced() * 1. / self.org_flops  # reduced
        self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
        self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
        obs = self.layer_embedding[self.cur_ind, :].copy()

        return obs, reward, done, info_set

    def reset(self):
        # restore env by loading the checkpoint
        # self.model.load_state_dict(self.checkpoint)
        self.model = copy.deepcopy(self.backup_model)
        self.cur_ind = 0 
        self.strategy = []  # pruning strategy
        self.d_prime_list = []
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)
        # reset layer embeddings
        self.layer_embedding[:, -1] = 1.
        self.layer_embedding[:, -2] = 0.
        self.layer_embedding[:, -3] = 0.
        obs = self.layer_embedding[0].copy()
        obs[-2] = sum(self.wsize_list[1:]) * 1. / sum(self.wsize_list)
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0
        # for share index
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}
        self.reduced_params = []
        return obs

    def set_export_path(self, path):
        self.export_path = path

    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None):
        m_list = list(self.model.modules())
        op = m_list[op_idx]
        assert (preserve_ratio <= 1.)

        if preserve_ratio == 1:  # do not prune
            return 1., op.weight.size(1), None  

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        n, c = op.weight.size(0), op.weight.size(1) 
        d_prime = format_rank(c * preserve_ratio) 
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > c:
            d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round) # floor向下取整

        extract_t1 = time.time()
        if self.use_new_input:  # this is slow and may lead to overfitting
            self._regenerate_input_feature()

        X = self.layer_info_dict[op_idx]['input_feat'] 
        Y = self.layer_info_dict[op_idx]['output_feat'] 

        weight = op.weight.data.cpu().numpy() 
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]
        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1
        fit_t1 = time.time()

        if preserve_idx is None:  
            importance = np.abs(weight).sum((0, 2, 3)) # importance.shape = <class 'tuple'>: (32,)
            sorted_idx = np.argsort(-importance) 
            preserve_idx = sorted_idx[:d_prime]  
        assert len(preserve_idx) == d_prime
        mask = np.zeros(weight.shape[1], bool) # mask.shape =  <class 'tuple'>: (32,)
        mask[preserve_idx] = True

        # reconstruct, X, Y <= [N, C]
        masked_X = X[:, mask] 
        if weight.shape[2] == 1:  # 1x1 conv or fc
            from lib.utils import least_square_sklearn
            rec_weight = least_square_sklearn(X=masked_X, Y=Y) # Y.shape = <class 'tuple'>: (30000, 64)
            rec_weight = rec_weight.reshape(-1, 1, 1, d_prime)  # rec_weight = (C_out, K_h, K_w, C_in') = (64, 1, 1, 16)
            rec_weight = np.transpose(rec_weight, (0, 3, 1, 2))  # rec_weight = (C_out, C_in', K_h, K_w) = (64, 16, 1, 1)
        else:
            raise NotImplementedError('Current code only supports 1x1 conv now!')
        if not self.export_model:  # pad, pseudo compress
            rec_weight_pad = np.zeros_like(weight) # rec_weight_pad = <class 'tuple'>: (64, 32, 1, 1)
            rec_weight_pad[:, mask, :, :] = rec_weight
            rec_weight = rec_weight_pad 

        if op_type == 'Linear':
            rec_weight = rec_weight.squeeze()
            assert len(rec_weight.shape) == 2
        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1

        op.weight.data = torch.from_numpy(rec_weight).cuda() 
        action = np.sum(mask) * 1. / len(mask)  

        if self.export_model:  
            prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx) - 1] # prune previous buffer ops
            for idx in range(prev_idx, op_idx): 
                m = m_list[idx]
                if type(m) == nn.Conv2d:  # depth-wise
                    tensor_weight = torch.from_numpy(m.weight.data.cpu().numpy()) # tensor_weight = torch.Size([32, 1, 3, 3]);
                    temp_weight = tensor_weight[mask, :, :, :].cuda() # temp_weight = torch.Size([32, 1, 3, 3])
                    m.weight.data = temp_weight 
                    if m.groups == m.in_channels:
                        m.groups = int(np.sum(mask)) 
                elif type(m) == nn.BatchNorm2d: 
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda()
                    m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
                    m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
                    m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()
        
        return action, d_prime, preserve_idx

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        for i, idx in enumerate(self.prunable_idx): 
            flop = self.layer_info_dict[idx]['flops'] 
            buffer_flop = self._get_buffer_flops(idx) 

            if i == self.cur_ind - 1:  # TODO: add other member in the set
                this_comp += flop * self.strategy_dict[idx][0] # 
                # add buffer (but not influenced by ratio)
                other_comp += buffer_flop * self.strategy_dict[idx][0]
            elif i == self.cur_ind:
                this_comp += flop * self.strategy_dict[idx][1]
                # also add buffer here (influenced by ratio)
                this_comp += buffer_flop
            else:
                other_comp += flop * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
                # add buffer
                other_comp += buffer_flop * self.strategy_dict[idx][0]  # only consider input reduction

        self.expected_min_preserve = other_comp + this_comp * action
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be)

        return action

    def _get_buffer_flops(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _cur_flops(self):
        flops = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            flops += self.layer_info_dict[idx]['flops'] * c * n
            # add buffer computation
            flops += self._get_buffer_flops(idx) * c  # only related to input channel reduction
        return flops

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    def _init_data(self):
        val_size = 5000 if 'cifar' in self.data_type else 3000 #  or 'Flower' 
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        use_real_val=self.use_real_val,
                                                                        shuffle=False)  # same sampling
        if self.use_real_val:  # use the real val set for eval, which is actually wrong
            print('*** USE REAL VALIDATION SET!')

    def _build_index(self):
        self.prunable_idx = [] 
        self.prunable_ops = [] 
        self.layer_type_dict = {} 
        self.strategy_dict = {} 
        self.buffer_dict = {} 
        this_buffer_list = []
        self.org_channels = [] 
        # build index and the min strategy dict
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.prunable_layer_types:
                if type(m) == nn.Conv2d and m.groups == m.in_channels: 
                    this_buffer_list.append(i)
                else: 
                    self.prunable_idx.append(i)
                    self.prunable_ops.append(m) 
                    self.layer_type_dict[i] = type(m)
                    self.buffer_dict[i] = this_buffer_list
                    this_buffer_list = []  # empty
                    self.org_channels.append(m.in_channels if type(m) == nn.Conv2d else m.in_features) # common conv or linear layer

                    self.strategy_dict[i] = [self.lbound, self.lbound] 

        self.strategy_dict[self.prunable_idx[0]][0] = 1  
        self.strategy_dict[self.prunable_idx[-1]][1] = 1  

        self.shared_idx = []
        if self.model == 'mobilenetv2': 
            connected_idx = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # to be partitioned
            last_ch = -1
            share_group = None
            for c_idx in connected_idx:
                if self.prunable_ops[c_idx].in_channels != last_ch:  # new group
                    last_ch = self.prunable_ops[c_idx].in_channels
                    if share_group is not None:
                        self.shared_idx.append(share_group)
                    share_group = [c_idx]
                else:  # same group
                    share_group.append(c_idx)
            self.shared_idx.append(share_group)
            print('=> Conv layers to share channels: {}'.format(self.shared_idx))

        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)

        self.buffer_idx = []
        for k, v in self.buffer_dict.items():
            self.buffer_idx += v

        print('=> Prunable layer idx: {}'.format(self.prunable_idx))
        print('=> Buffer layer idx: {}'.format(self.buffer_idx))
        print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}

    def _cur_params(self):
        m_list = list(self.model.modules())
        pruned_wsize_list = []
        
        from lib.utils import measure_layer_for_pruning
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        for idx in self.prunable_idx + self.buffer_idx:
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)

        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
                self.data_saver.append((input.clone(), target.clone()))
                input_var = torch.autograd.Variable(input).cuda()
                # inference and collect stats
                _ = self.model(input_var)
                if i_b == 0: 
                    for idx in self.prunable_idx + self.buffer_idx:
                        pruned_wsize_list.append(m_list[idx].params)
                else:
                    break

        return sum(pruned_wsize_list) + self.params_bn_size


    def _extract_layer_information(self):
        m_list = list(self.model.modules())

        self.data_saver = []
        self.layer_info_dict = dict()
        self.wsize_list = []
        self.flops_list = []

        from lib.utils import measure_layer_for_pruning

        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        for idx in self.prunable_idx + self.buffer_idx:  
            m = m_list[idx]
            m.old_forward = m.forward 
            m.forward = new_forward(m) 

        # extract information through inference, such as, params, flops.
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                if input.shape[0] != self.train_loader.batch_size:
                    break
                self.data_saver.append((input.clone(), target.clone()))
                input_var = torch.autograd.Variable(input).cuda()

                _ = self.model(input_var)

                if i_b == 0:  
                    for idx in self.prunable_idx + self.buffer_idx:
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        self.wsize_list.append(m_list[idx].params)
                        self.flops_list.append(m_list[idx].flops)
                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    f_out_np = m_list[idx].output_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save, f_out2save = None, None
                        elif m_list[idx].weight.size(3) > 1:  # normal conv
                            f_in2save, f_out2save = f_in_np, f_out_np
                        else:  
                            # assert f_out_np.shape[2] == f_in_np.shape[2]  
                            randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
                            randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)
                            # input: [N, C, H, W]
                            self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                            self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

                            # f_in_np.shape = <class 'tuple'>: (50, 64, 8, 8)
                            # f_in2save = <class 'tuple'>: (500, 64)
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)

                            f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                        f_out2save = f_out_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                        self.layer_info_dict[idx]['output_feat'] = f_out2save
                    else:
                        # np.concatenate(axis=0) 等价于 np.vstack()
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save)) 
                        self.layer_info_dict[idx]['output_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['output_feat'], f_out2save))

    def _regenerate_input_feature(self):
        # only re-generate the input feature
        m_list = list(self.model.modules())

        # delete old features
        for k, v in self.layer_info_dict.items():
            if 'input_feat' in v:
                v.pop('input_feat')

        # now let the image flow
        print('=> Regenerate features...')

        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.data_saver):
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save = None
                        else:
                            randx = self.layer_info_dict[idx][(i_b, 'randx')]
                            randy = self.layer_info_dict[idx][(i_b, 'randy')]
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:  # fc
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))

    def _build_state_embedding(self):
        layer_embedding = []
        module_list = list(self.model.modules())
        for i, ind in enumerate(self.prunable_idx):
            m = module_list[ind]
            this_state = []
            this_state.append(self.preserve_ratio) 
            if type(m) == nn.Conv2d:
                this_state.append(i)  # index
                this_state.append(0)  # layer type, 0 for conv
                this_state.append(m.in_channels)  # in channels
                this_state.append(m.out_channels)  # out channels
                this_state.append(m.stride[0])  # stride
                this_state.append(m.kernel_size[0])  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size  
            elif type(m) == nn.Linear:
                this_state.append(i)  # index
                this_state.append(1)  # layer type, 1 for fc
                this_state.append(m.in_features)  # in channels
                this_state.append(m.out_features)  # out channels
                this_state.append(0)  # stride
                this_state.append(1)  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size

            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))

        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape

        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def _validate(self, val_loader, model, verbose=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        t1 = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
                  (losses.avg, top1.avg, top5.avg, t2 - t1))
        if self.acc_metric == 'acc1':
            return top1.avg
        elif self.acc_metric == 'acc5':
            return top5.avg
        else:
            raise NotImplementedError


    def render(self):
        pass


    def _seed(self, seed):
        # random.seed(seed) 
        foo = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        random_ratio = choice(foo)
        if self.env_index == 0:
            fixed_ratio = 0.65
        elif self.env_index == 1:
            fixed_ratio = 0.7
        elif self.env_index == 2:
            fixed_ratio = 0.75
        elif self.env_index == 3:
            fixed_ratio = 0.8
        # fixed_ratio = 0.8

        preserve_ratio = np.clip(fixed_ratio, self.lbound, self.rbound)
        return preserve_ratio

    def _get_model_and_checkpoint(self, model, dataset, checkpoint_path, n_gpu=1):
        if model == 'mobilenet' and dataset == 'imagenet':
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=1000)
        elif model == 'mobilenetv2' and dataset == 'imagenet':
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=1000)
        elif model == 'resnet50' and dataset == 'cifar10':
            net = models.resnet50(pretrained=False)
            net.load_state_dict(torch.load('checkpoints/resnet50-19c8e357.pth'))
            net = net.cuda()
            if n_gpu > 1:
                net = torch.nn.DataParallel(net, range(n_gpu))
            return net, deepcopy(net.state_dict())
        elif model == 'mobilenet' and dataset == 'cifar10':
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=10)
        elif model == "vgg16" and dataset == 'cifar10':
            from models.vgg_cifar import VGG
            net = VGG(vgg_name="vgg16", num_classes=10)
        elif model == 'mobilenetv2' and dataset == 'cifar10':
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=10)
        elif model == 'mobilenet' and (dataset == 'MNIST' or dataset == 'FashionMNIST'):
            from models.mobilenet import MobileNetMNIST
            net = MobileNetMNIST(n_class=102) 
        elif model == "mobilenet" and dataset == "Flower":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=102)
        elif model == "mobilenet" and dataset == "cub200":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=200)
        else:
            raise NotImplementedError

        root_path = os.path.abspath(os.path.dirname(__file__)) + "/../"
        sd = torch.load(root_path + checkpoint_path)
        if 'tar' in checkpoint_path: 
            if 'state_dict' in sd: 
                sd = sd['state_dict']
        else:
            sd = sd['net']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        net.load_state_dict(sd)

        net = net.cuda()
        if n_gpu > 1:
            net = torch.nn.DataParallel(net, range(n_gpu))
        return net, deepcopy(net.state_dict())
