# Code for "Conditional Automated Channel Pruning for Deep Neural Networks"
# implement by Yixin Liu
# seyixinliu@mail.scut.edu.cn

import os
import numpy as np
import argparse
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True

from env.cacp_channel_pruning_env import ChannelPruningEnv as CACPChannelPruningEnv
from lib.agent import DDPG
from lib.utils import get_output_folder
import timeit

from tensorboardX import SummaryWriter
from utils import _get_model_and_checkpoint


parser = argparse.ArgumentParser(description='CACP search script')

parser.add_argument('--job', default='search', type=str, help='support option: search')
parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
# env
parser.add_argument('--model', default='vgg16', type=str, help='model to prune')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar/imagenet)')
parser.add_argument('--data_root', default='./dataset/cifar10', type=str, help='dataset path')
# parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
parser.add_argument('--compression_targets', default='0.6,0.7',  type=str)

parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
parser.add_argument('--reward', default='conditional1', type=str, help='Setting the reward')
parser.add_argument('--acc_metric', default='acc1', type=str, help='use acc1 or acc5')
parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')
# ddpg
parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
parser.add_argument('--warmup', default=30, type=int,
                    help='time without training but only filling the replay memory')
parser.add_argument('--discount', default=1., type=float, help='')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--rmsize', default=30, type=int, help='memory size for each layer')
parser.add_argument('--window_length', default=1, type=int, help='')
parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
# noise (truncated normal distribution)
parser.add_argument('--init_delta', default=0.5, type=float,
                    help='initial variance of truncated normal distribution')
parser.add_argument('--delta_decay', default=0.95, type=float,
                    help='delta decay during exploration')

# fm-reconstruction
parser.add_argument('--repair_batchs', default=3, type=int, help='fm-reconstruction')
parser.add_argument('--repair_points', default=6, type=int, help='fm-reconstruction')

# training
parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
parser.add_argument('--output', default='./cacp_logs', type=str, help='')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--init_w', default=0.003, type=float, help='')
parser.add_argument('--train_episode', default=160, type=int, help='train iters each timestep')
parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
parser.add_argument('--seed', default=2022, type=int, help='random seed to set')
parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
parser.add_argument('--n_worker', default=8, type=int, help='number of data loader worker')
parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')
parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')

args = parser.parse_args()


def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
    if model == 'vgg16' and dataset == 'cifar10':
        from models.vgg_cifar import VGG
        net = VGG(vgg_name='vgg16')
    else:
        raise NotImplementedError
    sd = torch.load(checkpoint_path)
    if 'state_dict' in sd:  # a checkpoint but not a state_dict
        sd = sd['state_dict']
    if 'net' in sd:  # a checkpoint but not a state_dict
        sd = sd['net']
    net.load_state_dict({k.replace('module.',''):v for k,v in sd.items()})
    net = net.cuda()
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, range(n_gpu))
    return net, deepcopy(net.state_dict())


def train(num_episode, agent, env, output):
    timer = timeit.default_timer
    start_time = timer()

    agent.is_training = True

    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    reward_curve = []
    
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None: # this mean we start from the first layer
            observation = deepcopy(env.reset()) # reset and sample new beta
            agent.reset(observation)

        
        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
            # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermideate model
        # if episode % int(num_episode / 3) == 0:
        #     agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            print('#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f},strategy: {},d_prime: {}, time: {:.4f}'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio'],info['strategy'],info['d_prime'], (timer() - start_time)/60))
            text_writer.write(
                '#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f},strategy: {},d_prime: {}, time: {:.4f}\n'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio'],info['strategy'],info['d_prime'], (timer() - start_time)/60))
            
            final_reward = T[-1][0]
            reward_curve.append([episode,env.beta,info['accuracy']*0.01])
            # print('final_reward: {}'.format(final_reward))
            # agent observe and update policy
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    agent.update_policy()

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            tfwriter.add_scalar(f'info/reward_{env.beta}', final_reward, episode)
            tfwriter.add_scalar(f'info/best_reward_{env.beta}', env.best_reward[env.cur_beta_idx], episode)
            tfwriter.add_scalar(f'info/accuracy_{env.beta}', info['accuracy'], episode)
            tfwriter.add_scalar(f'info/compress_ratio_{env.beta}', info['compress_ratio'], episode)
            tfwriter.add_text(f'info/best_policy_{env.beta}', str(env.best_strategy[env.cur_beta_idx]), episode)

            
            # tfwriter.close()
            # record the preserve rate for each layer
            # for i, preserve_rate in enumerate(env.strategy):
            #     tfwriter.add_scalar(f'preserve_rate4each_layer/{env.beta}', preserve_rate, episode)
            # text_writer.close()
            # text_writer.write('best reward: {}\n'.format(env.best_reward))
            # text_writer.write('best policy: {}\n'.format(env.best_strategy))
    # text_writer.close()
    np.save("acc_curve_conditional2.npy",reward_curve)


if __name__ == "__main__":

    def split_targets(a):
        b = [float(i) for i in a.split(',') ]
        return b
    
    args.compression_targets = split_targets(args.compression_targets)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    model, checkpoint = _get_model_and_checkpoint(args.model, args.dataset, checkpoint_path=args.ckpt_path,
                                                 n_gpu=args.n_gpu)


    env = CACPChannelPruningEnv(args.model, model, checkpoint, args.dataset,
                            args.compression_targets,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args)

    if args.job == 'search':
        # build folder and logs
        base_folder_name = '{}_{}_search'.format(args.model, args.dataset)
        if args.suffix is not None:
            base_folder_name = base_folder_name + '_' + args.suffix
        args.output = get_output_folder(args.output, base_folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tfwriter = SummaryWriter(logdir=args.output)
        text_writer = open(os.path.join(args.output, 'log.txt'), 'w')
        print('=> Output path: {}...'.format(args.output))

        nb_states = env.layer_embedding.shape[1]
        nb_actions = 1  # just 1 action here

        args.rmsize = args.rmsize * len(env.prunable_idx)  # for each layer
        print('** Actual replay buffer size: {}'.format(args.rmsize))

        agent = DDPG(nb_states, nb_actions, args)
        train(args.train_episode, agent, env, args.output)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))