import csv
import os
from functools import partial

import numpy as np
from components.episode_buffer import EpisodeBatch
# from envs import REGISTRY as env_REGISTRY
import magent

def get_config_pursuit_attack(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    # -0.3 -0.15 0
    predator = cfg.register_agent_type("predator", {'width': 1, 'length': 1, 'hp': 1, 'speed': 1, 'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(1), 'attack_penalty': -0.15})

    prey = cfg.register_agent_type("prey", {'width': 1, 'length': 1, 'hp': 1, 'speed': 0, 'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)})

    predator_group = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    b = gw.AgentSymbol(predator_group, index='any')
    c = gw.AgentSymbol(prey_group, index='any')

    # tigers get reward when they attack a deer simultaneously
    e1 = gw.Event(a, 'attack', c)
    e2 = gw.Event(b, 'attack', c)
    cfg.add_reward_rule(e1 & e2, receiver=[a, b], value=[0.5, 0.5])

    # a = gw.AgentSymbol(predator_group, index='any')
    # b = gw.AgentSymbol(prey_group, index='any')

    # cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[1, -1])

    return cfg

class EpisodeRunner:

    def __init__(self, args, logger):
        # print('EpisodeRunner init')
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        cfg, args = get_config_pursuit_attack(args)
        env = magent.GridWorld(cfg)
        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env = env
        self.episode_limit = self.env.episode_limit
        map_name = self.args.env_args['map_name']
        seed = self.args.env_args['seed']
        self.csv_dir = f'./qplex_sdq_intrinsic_anneal_csv_files/{map_name}/'
        self.csv_path = f'{self.csv_dir}seed_{seed}.csv'
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1, preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def writereward(self, path, reward, win_rate, step):
        if os.path.isfile(path):
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, reward, win_rate])
        else:
            with open(path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'reward', 'win_rate'])
                csv_write.writerow([step, reward, win_rate])

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {"state": [self.env.get_state()], "avail_actions": [
            self.env.get_avail_actions()], "obs": [self.env.get_obs()]}
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)
                          for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self.writereward(self.csv_path, np.mean(
                cur_returns), cur_stats['battle_won'] / cur_stats['n_episodes'], self.t_env)
            print('=' * 30)
            print('mean_return', np.mean(cur_returns), 't_env', self.t_env)
            print(cur_stats)
            print('=' * 30)
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean",
                             np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std",
                             np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v / stats["n_episodes"], self.t_env)
        stats.clear()
