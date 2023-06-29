import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing
import csv
os.environ["OMP_NUM_THREADS"] = "1" 
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
import torch.distributions as tdist
from core import environment
from core import file_writer
from core import prof
from core import vtrace
from env.channel_pruning_env_mobilenet import MobileNetAutoPruningEnv
from env.channel_pruning_env_vgg import VGGNetAutoPruningEnv
from env.channel_pruning_env_vgg16 import ConvNetAutoPruningEnv
from env.channel_pruning_env_vgg16_backup import BackupConvNetAutoPruningEnv


os.environ["CUDA_VISIBLE_DEVICES"] = "0" # FIXME
logging.getLogger('PIL').setLevel(logging.WARNING)

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="AutoPrune-v0",
                    help="auto channel pruning environment.") # PongNoFrameskip-v4
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
cur_path =  os.path.abspath(os.path.dirname(__file__))
parser.add_argument("--savedir", default=cur_path + "/logs/torchbeast/autoprune", 
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=1, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=10000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=4, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=100, type=int, metavar="T", # 180
                    help="The unroll length (time dimension).") 
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--seed", default=2022, type=int,
                    metavar="S", help="Seed")
parser.add_argument("--num_learner_threads", "--num_threads", default=1, type=int,
                    metavar="N", help="Number learner threads.") 
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--skip", default=None, type=int, metavar="Frame", help="Whether frame skip.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.01, # 0.0006 
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="none",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048, # 0.003, # 0.00048  lr=0.0001
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0.5, type=float, 
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]] 

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_entropy_loss(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)

def compute_entropy_loss_continous(logits):
    logits_flatten = torch.flatten(logits, start_dim=0, end_dim=1)
    dist = tdist.normal.Normal(logits_flatten[:, 0], logits_flatten[:, 1])
    entropys = dist.entropy()
    return torch.sum(entropys)

def compute_policy_gradient_loss_continuous(logits, actions, advantages):
    logits_flatten = torch.flatten(logits, 0, 1)
    actions = torch.flatten(actions)
    dist = tdist.normal.Normal(logits_flatten[:, 0], logits_flatten[:, 1])
    cross_entropy = -1 * dist.log_prob(actions)
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())

def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, start_dim=0, end_dim=1), dim=-1), 
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
        flags, 
        actor_index: int,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        model: torch.nn.Module,
        buffers: Buffers,
        initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        # temp_env = MobileNetAutoPruningEnv(actor_index, flags.xpid)
        temp_env = BackupConvNetAutoPruningEnv(actor_index, flags.xpid, flags.num_actors)
        # temp_env = ConvNetAutoPruningEnv(actor_index, flags.xpid)
        # temp_env = create_env(flags)
        # seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        # temp_env.seed(seed)

        env = environment.Environment(temp_env)
        env_output = env.initial() # s_0, r_(-1)

        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)  # a_0, v_0

        while True:
            index = free_queue.get()
            if index is None:
                break

            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state) 

                timings.time("model")

                env_output = env.step(agent_output["action"]) 

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
        flags,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        buffers: Buffers,
        initial_agent_state_buffers,
        timings,
        lock=threading.Lock(), 
):
    with lock:
        timings.time("lock") 
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = { # batch['frame'] = torch.Size([81, 8, 4, 84, 84]); batch['reward'] = torch.Size([81, 8]); batch['action'] = torch.Size([81, 8])
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers 
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
        flags,
        actor_model,
        model,
        batch,
        initial_agent_state,
        optimizer,
        scheduler,
        lock=threading.Lock(),# noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)  

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1) 
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting # batch["done"] dtype=bool

        # 用 vtrace 计算 advantage
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        # pg_loss = compute_policy_gradient_loss(
        #     learner_outputs["policy_logits"],
        #     batch["action"],
        #     vtrace_returns.pg_advantages,
        # )
        # baseline_loss = flags.baseline_cost * compute_baseline_loss(
        #     vtrace_returns.vs - learner_outputs["baseline"] # vs 是状态 s 的 Target Value
        # )
        # entropy_loss = flags.entropy_cost * compute_entropy_loss(
        #     learner_outputs["policy_logits"]
        # ) # baseline_loss, entropy_loss, pg_loss.shape = []，因为标量没有shape；pg_loss = tensor(-511.5512)
        
        pg_loss = compute_policy_gradient_loss_continuous(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss_continous(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()  
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step() 
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags): 
    if flags.xpid is None:
        flags.xpid = "autoPrune-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    STATE_SPACE = 12 # 10 # 11 # 12 # 13
    model = Net(observation_shape=(1, STATE_SPACE), num_actions=1, use_lstm=flags.use_lstm)
    # env = create_env(flags)
    # model = Net(env.observation_space.shape, env.action_space.n, flags.use_lstm)


    # （1）auto channel env 的 replay buffers
    buffers = create_buffers(flags, obs_shape=(1, STATE_SPACE), num_actions=2)
    # （2）atari env 的 replay buffers
    # buffers = create_buffers(flags, env.observation_space.shape, model.num_actions)

    model.share_memory()

    initial_agent_state_buffers = [] 
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("spawn") # spawn
    free_queue = ctx.SimpleQueue() 
    full_queue = ctx.SimpleQueue()  
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(observation_shape=(1, STATE_SPACE), num_actions=1, use_lstm=flags.use_lstm).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon, 
        alpha=flags.alpha, 
    )

    def lr_lambda(epoch): 
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        # "mean_episode_step",
        # "episode_steps",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()

            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers, 
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler,
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)


    def test_traning(flags, num_episodes: int = 5):
        returns = []
        while len(returns) < num_episodes:
            gym_env = MobileNetAutoPruningEnv("TODO")
            env = environment.Environment(gym_env)
            observation = env.initial()

            model = Net(observation_shape=(1, STATE_SPACE), num_actions=1, use_lstm=flags.use_lstm)
            model.eval()
            model.load_state_dict(learner_model.state_dict()) 
            agent_state = model.initial_state(batch_size=1)

            if flags.mode == "test_render":
                env.gym_env.render()

            while True:
                agent_outputs, agent_state = model(observation, agent_state)
                observation = env.step(agent_outputs["action"])

                if observation["done"].item():
                    returns.append(observation["episode_return"].item())
                    break

            env.close()

        with open(os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "test_training.csv")), "a") as f:
            writer = csv.writer(f)
            writer.writerow([step, num_episodes, sum(returns) / len(returns)])


    def start_test(flags):
        nonlocal step
        count = 1
        with open(os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "test_training.csv")), "a") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "average_step", "averate_episode_return"])

        while step < flags.total_steps:
            if step > 25000 * count:
                count += 1
                # eps **= count
                test_traning(flags=flags, num_episodes=5)


    # test_thread = threading.Thread(
    #     target=start_test, name="test-thread", args=(flags,)
    # )
    # test_thread.start()
    # threads.append(test_thread)


    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )
    stat_keys_console = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        # "mean_episode_step",
        "episode_returns",
    ]

    timer = timeit.default_timer
    lock = threading.Lock()
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            # time.sleep(5)
            # time.sleep(60 * 10) 
            
            if len(stats) is not 0:
                with lock:
                    if timer() - last_checkpoint_time > 10 * 60: # 10 * 20:  # Save every 10 min = 10 * 60.
                        checkpoint()
                        last_checkpoint_time = timer()

                    sps = (step - start_step) / (timer() - start_time)
                    if stats.get("episode_returns", None):
                        mean_return = (
                                "Return per episode: %.1f. " % stats["mean_episode_return"]
                        )
                    else:
                        mean_return = ""
                    total_loss = stats.get("total_loss", float("inf"))

                    stats_console = dict(step=step)
                    stats_console.update({k: stats[k] for k in stat_keys_console})
                    logging.info(
                        "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                        step,
                        sps,
                        total_loss,
                        mean_return,
                        pprint.pformat(stats_console),
                    )
                    stats = {}

    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()



# def test(flags, num_episodes: int = 1):
#     if flags.xpid is None:
#         checkpointpath = "./latest/model.tar"
#     else:
#         checkpointpath = os.path.expandvars(
#             os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
#         )
#
#     gym_env = gym.make(flags.env)
#     env = environment.Environment(gym_env)
#     observation = env.initial_uav()
#
#     model = Net((4, 63, 63), gym_env.action_space.n, flags.use_lstm)
#     model.eval()
#     checkpoint = torch.load(checkpointpath, map_location="cpu")
#     model.load_state_dict(checkpoint["model_state_dict"]) 
#     agent_state = model.initial_state(batch_size=1)
#
#     returns = []
#
#     while len(returns) < num_episodes:
#         if flags.mode == "test_render":
#             env.gym_env.render()
#
#         agent_outputs, agent_state = model(observation, agent_state)
#         observation = env.step_uav(agent_outputs["action"])
#
#         if observation["done"].item():
#             returns.append(observation["episode_return"].item())
#             logging.info(
#                 "Episode ended after %d steps. Return: %.1f",
#                 observation["episode_step"].item(),
#                 observation["episode_return"].item(),
#             )
#
#     env.close()
#     logging.info(
#         "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
#     )


class AutoPruneNet(nn.Module):

    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AutoPruneNet, self).__init__()
        self.use_lstm = use_lstm
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        input_layer = 64# 100
        hidden1 = 128 # 400
        hidden2 = 64 # 300
        self.input_layer = nn.Linear(observation_shape[1], input_layer)
        self.fc1 = nn.Linear(input_layer, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        core_output_size = self.fc2.out_features + num_actions + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.mu = nn.Linear(core_output_size, 1)
        self.sigma = nn.Linear(core_output_size, 1)
        self.baseline = nn.Linear(core_output_size, 1)


    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].  T = Time, B = Batch
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1) 
        x = x.float()
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(T * B, -1)

        last_action = inputs["last_action"].view(T * B, 1).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1).float()

        core_input = torch.cat([x, clipped_reward, last_action], dim=-1)

        if self.use_lstm and len(core_state) is not 0:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        mu = (torch.sigmoid(self.mu(core_output)))
        sigma = F.softplus(self.sigma(core_output)) + 0.001
        policy_logits = torch.cat((mu, sigma), -1)
        baseline = self.baseline(core_output)
        action = torch.zeros(policy_logits.shape[0], 1)
        dist = tdist.Normal(mu, sigma)
        action = (torch.tanh(dist.sample()) + 1) / 2 
        
        policy_logits = policy_logits.view(T, B, 2)
        baseline = baseline.view(T, B) # baseline.shape = torch.Size([1, 1])
        action = action.view(T, B) # action.shape = torch.Size([1, 1])

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action), # policy_logits.type=torch.float32; action.type=torch.int64
            core_state,
        )

class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

Net = AutoPruneNet

def create_env(flags):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(flags.env),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
            skip=flags.skip
        )
    )

def main(flags):
    if flags.mode == "train":
        train(flags)
    # else:
        # test(flags)

if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
