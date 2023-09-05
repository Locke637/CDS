import csv
import os
from functools import partial

import numpy as np
from components.episode_buffer import EpisodeBatch
# from envs import REGISTRY as env_REGISTRY
import magent
import wandb
import os
import random
import gym
import sys

sys.path.append("/home/sqliu/CDS")
# sys.path.append("/home/sqliu/CDS/CDS_overcooked")
# from PantheonRL.overcookedgym.overcooked_utils import LAYOUT_LIST
from CDS_overcooked.PantheonRL.overcookedgym.overcooked import OvercookedMultiEnv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# wandb.config.update(args)


class EpisodeRunner:

    def __init__(self, args, logger):
        # print('EpisodeRunner init')
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        # unident_s random3 scenario4 random0 scenario3 homo homo_hard five_by_five
        self.args.env_name = 'five_by_five'
        env = OvercookedMultiEnv(self.args.env_name)
        # if args.env_name == 'overcooked_simple':
        #     env = OvercookedMultiEnv("simple")
        #     # env = gym.make('OvercookedMultiEnv-v0', {"layout_name":"simple"})
        #     # _, _ = env.reset()
        # elif args.env_name == 'five_by_five':
        #     env = OvercookedMultiEnv("five_by_five")
        # elif args.env_name == 'random1':
        #     env = OvercookedMultiEnv("random1")
        # Exception('env name error')
        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env = env

        self.n_agents = 2
        # args.n_actions = env.get_total_actions()
        # print(env.action_space.n)
        self.n_actions = env.action_space.n
        # print(env.observation_space.shape[0])
        self.state_shape = env.observation_space.shape[0] * self.n_agents
        self.act_dim = self.n_actions
        # args.obs_shape = env.get_obs_size()
        self.obs_shape = env.observation_space.shape[0]
        self.episode_limit = 100
        # self.penalty = -0.05
        self.coding = False # True False
        if not self.coding:
            name = 'cds_' + self.args.env_name + '_re'
            wandb.init(project="hierarchical MARL", entity="637-muiltagent", name=name, notes='-0.2')
        map_name = self.args.env_args['map_name']
        seed = self.args.env_args['seed']
        self.csv_dir = f'./qplex_sdq_intrinsic_anneal_csv_files/{map_name}/'
        self.csv_path = f'{self.csv_dir}seed_{seed}.csv'
        self.t = 0

        self.t_env = 0
        self.epoch = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1, preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        env_info = {}
        env_info["episode_limit"] = self.episode_limit
        env_info["n_agents"] = self.n_agents
        env_info["n_actions"] = self.n_actions
        env_info["state_shape"] = self.state_shape
        # args.unit_dim = runner.env.get_unit_dim()
        env_info["obs_shape"] = self.obs_shape
        return env_info

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        next_state = self.env.multi_reset()
        self.t = 0
        return next_state

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
        next_state = self.reset()

        terminated = False
        episode_return = 0
        step = 0
        episode_reward = 0
        win_tag = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            # obs_all = self.env.get_observation(handles[0])
            # fixed_obs_all = self.env.get_observation(handles[1])
            # view = obs_all[0]
            # feature = obs_all[1]
            # fixed_view = fixed_obs_all[0]
            # fixed_feature = fixed_obs_all[1]
            # obs = []
            # fixed_obs = []
            # state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()
            # # print(self.n_agents, self.env.get_num(handles[0]))

            # for j in range(self.n_agents):
            #     obs.append(np.concatenate([view[j].flatten(), feature[j]]))
            #     # fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))

            obs = next_state
            state = np.concatenate([obs[0], obs[1]])

            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_action = np.ones(self.n_actions)
                avail_actions.append(avail_action)

            pre_transition_data = {"state": [state], "avail_actions": np.array(avail_actions), "obs": [obs]}

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            tmp_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # actions = []

            # for i, action in enumerate(tmp_actions):
            #     # print(action.type)
            #     if isinstance(action, np.int64) or isinstance(action, int):
            #         actions.append(action.astype(np.int32))
            #     else:
            #         action = action.cpu()
            #         actions.append(action.numpy().astype(np.int32))

            # print(actions)
            actions = tmp_actions[0]
            # print(actions)
            next_state, reward, terminated, info = self.env.multi_step(actions[0], actions[1])
            reward = sum(reward)
            # penalty_num = 0
            # # print(actions)
            # for a in actions:
            #     if a == self.n_actions - 1:
            #         penalty_num += 1
            # reward += self.penalty * penalty_num
            if step == self.episode_limit - 1:
                terminated = 1.

            # reward, terminated, env_info = self.env.step(actions[0])
            # episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward, )],
                "terminated": [(terminated, )],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            step += 1
            episode_reward += reward

        obs = next_state
        state = np.concatenate([obs[0], obs[1]])

        last_data = {"state": [state], "avail_actions": [np.ones((self.n_agents, self.n_actions))], "obs": [obs]}

        # last_data = {"state": [self.env.get_state()], "avail_actions": [self.env.get_avail_actions()], "obs": [self.env.get_obs()]}
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.epoch += 1

        cur_returns.append(episode_return)
        # if not self.args.env_name == 'bridge_base':
        #     win_tag = win_tag / step / (self.n_agents + self.args.more_enemy)
        # else:
        #     win_tag = win_tag / self.n_agents
        episode_reward = round(episode_reward, 2)

        if test_mode:
            # self.writereward(self.csv_path, np.mean(cur_returns), cur_stats['battle_won'] / cur_stats['n_episodes'], self.t_env)
            # print('=' * 30)
            # print('mean_return', np.mean(cur_returns), 't_env', self.t_env)
            # print(cur_stats)
            # print('=' * 30)
            if not self.coding:
                wandb.log({"overcooked_recent_return": episode_reward}, step=self.epoch)
                # wandb.log({"win rate": win_tag}, step=self.epoch)
                # wandb.log({"win rate": rate}, step=epoch)
            # add time to print
            print('reward', episode_reward, 't_env', self.t_env, 'win_tag', win_tag)
            # self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            # self._log(cur_returns, cur_stats, log_prefix)
            # if hasattr(self.mac.action_selector, "epsilon"):
            #     self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

    def generate_map(self, handles):
        width = self.args.map_size
        height = self.args.map_size

        pos = []
        for y in range(height):
            if y > 3 and y < height - 4:
                for x in range(1, width - 1):
                    if x != width // 2:
                        pos.append((x, y))
        self.env.add_walls(pos=pos, method="custom")

        # pos = [(2, 2), (map_size - 3, 2), (map_size - 3, map_size - 3),  (2, map_size - 3)]
        # pos = [(self.args.map_size - 3, self.args.map_size - 3),  (2, self.args.map_size - 3), (2, 2), (self.args.map_size - 3, 2)]
        pos = []
        for i in range(self.n_agents):
            if i < 2:
                p = (np.random.randint(1, self.args.map_size - 2), np.random.randint(self.args.map_size - 4, self.args.map_size - 2))
                while p in pos:
                    p = (np.random.randint(1, self.args.map_size - 2), np.random.randint(self.args.map_size - 4, self.args.map_size - 2))
                # pos.append(p)
            else:
                p = (np.random.randint(1, self.args.map_size - 2), np.random.randint(1, 4))
                while p in pos:
                    p = (np.random.randint(1, self.args.map_size - 2), np.random.randint(1, 4))
                # pos.append(p)
            pos.append(p)
        self.env.add_agents(handles[0], method="custom", pos=pos)

        pos = [(self.args.map_size - 2, self.args.map_size - 2), (5, 5)]
        self.env.add_agents(handles[1], method="custom", pos=pos)

    def get_bridge_reward(self, pos):
        reward = 0
        for i in range(2):
            if pos[i][0] >= 1 and pos[i][0] <= self.args.map_size - 1 and pos[i][1] >= 1 and pos[i][1] <= 3:
                reward += 0.1 * (3 - pos[i][1])
            else:
                reward -= 0
            if pos[i][1] > 3:
                reward += 0.01 * (3 - pos[i][1])
        for i in range(2, self.n_agents):
            if pos[i][0] >= 1 and pos[i][0] <= self.args.map_size - 1 and pos[i][1] >= 7 and pos[i][1] <= self.args.map_size - 1:
                reward += 0.1 * (pos[i][1] - 7)
            else:
                reward -= 0
            if pos[i][1] < 7:
                reward += 0.01 * (pos[i][1] - 7)
        return reward

    def get_bridge_win_rate(self, pos):
        reward = np.zeros(self.n_agents)
        for i in range(2):
            if pos[i][0] >= 1 and pos[i][0] <= self.args.map_size - 1 and pos[i][1] >= 1 and pos[i][1] <= 3:
                reward[i] = 1
        for i in range(2, self.n_agents):
            if pos[i][0] >= 1 and pos[i][0] <= self.args.map_size - 1 and pos[i][1] >= 7 and pos[i][1] <= self.args.map_size - 1:
                reward[i] = 1
        return reward
