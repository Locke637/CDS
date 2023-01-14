import csv
import os
from functools import partial

import numpy as np
from components.episode_buffer import EpisodeBatch
# from envs import REGISTRY as env_REGISTRY
import magent
import wandb
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# wandb.config.update(args)


def get_config_double_attack_hard(map_size):
    gw = magent.gridworld
    cfg = gw.Config()
    # alg_args.map_size = 10  # 80 30
    # alg_args.n_agents = 4  # 6
    # alg_args.more_walls = 0
    # alg_args.more_enemy = -3
    # alg_args.random_num = 1
    # alg_args.mini_map_shape = 6  # battle:20 pursuit:30
    # map_size = alg_args.map_size

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type("predator", {'width': 1, 'length': 1, 'hp': 1, 'speed': 1, 'damage': 1, 'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(1), 'attack_penalty': 0})

    prey = cfg.register_agent_type("prey", {'width': 1, 'length': 1, 'hp': 2, 'step_recover': 2, 'speed': 1, 'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)})

    predator_group = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    b = gw.AgentSymbol(predator_group, index='any')
    c = gw.AgentSymbol(prey_group, index='any')

    # tigers get reward when they attack a deer simultaneously
    e1 = gw.Event(a, 'attack', c)
    e2 = gw.Event(b, 'attack', c)
    cfg.add_reward_rule(e1 & e2, receiver=[a, b, c], value=[0.75, 0.75, 1])

    # a = gw.AgentSymbol(predator_group, index='any')
    # b = gw.AgentSymbol(prey_group, index='any')

    # cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=[a, c], value=[1, -1])
    cfg.add_reward_rule(gw.Event(a, 'kill', c), receiver=[a, c], value=[-150, 0])
    cfg.add_reward_rule(gw.Event(b, 'kill', c), receiver=[b, c], value=[-150, 0])

    return cfg


def get_config_multi_target(map_size):
    gw = magent.gridworld
    cfg = gw.Config()
    # args.map_size = 15  # 80 30
    # args.n_agents = 4  # 6
    # args.more_walls = 0
    # args.more_enemy = 0
    # args.random_num = 1
    # args.mini_map_shape = 6  # battle:20 pursuit:30
    # map_size = args.map_size

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type("predator", {
        'width': 1,
        'length': 1,
        'hp': 1,
        'speed': 1,
        'damage': 1,
        'view_range': gw.CircleRange(5),
        'attack_range': gw.CircleRange(1),
        'attack_penalty': -0.1
    })

    prey = cfg.register_agent_type("prey", {'width': 1, 'length': 1, 'hp': 1, 'step_recover': 1, 'speed': 0, 'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)})

    predator_group = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    # b = gw.AgentSymbol(predator_group, index='any')
    c = gw.AgentSymbol(prey_group, 0)
    cmax = gw.AgentSymbol(prey_group, 1)
    b = gw.AgentSymbol(prey_group, 2)
    bmax = gw.AgentSymbol(prey_group, 3)

    # # tigers get reward when they attack a deer simultaneously
    # e1 = gw.Event(a, 'attack', c)
    # e2 = gw.Event(b, 'attack', c)
    # cfg.add_reward_rule(e1 & e2, receiver=[a, b], value=[0.75, 0.75])

    # a = gw.AgentSymbol(predator_group, index='any')
    # b = gw.AgentSymbol(prey_group, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=[a, c], value=[0.1, 1])
    cfg.add_reward_rule(gw.Event(a, 'kill', c), receiver=[a, c], value=[-10, 0])

    cfg.add_reward_rule(gw.Event(a, 'attack', cmax), receiver=[a, cmax], value=[0.2, 1])
    cfg.add_reward_rule(gw.Event(a, 'kill', cmax), receiver=[a, cmax], value=[-20, 0])

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[0.3, 1])
    cfg.add_reward_rule(gw.Event(a, 'kill', b), receiver=[a, b], value=[-30, 0])

    cfg.add_reward_rule(gw.Event(a, 'attack', bmax), receiver=[a, bmax], value=[0.4, 1])
    cfg.add_reward_rule(gw.Event(a, 'kill', bmax), receiver=[a, bmax], value=[-40, 0])

    return cfg


def get_config_bridge(map_size):
    gw = magent.gridworld
    cfg = gw.Config()
    # args.map_size = 11  # 80 30
    # args.n_agents = 4  # 6
    # args.more_walls = 1
    # args.more_enemy = -3
    # args.random_num = 1
    # args.mini_map_shape = 6  # battle:20 pursuit:30
    # map_size = args.map_size

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type("predator", {'width': 1, 'length': 1, 'hp': 1, 'speed': 1, 'damage': 2, 'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(1), 'attack_penalty': 0})

    prey = cfg.register_agent_type("prey", {'width': 1, 'length': 1, 'hp': 1, 'step_recover': 0, 'speed': 0, 'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)})

    predator_group = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)
    return cfg


def get_config_pursuit_attack(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    # -0.3 -0.15 0
    predator = cfg.register_agent_type("predator", {'width': 1, 'length': 1, 'hp': 1, 'speed': 1, 'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(1), 'attack_penalty': -0.2})

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


def get_config_pursuit_attack_ez(map_size):
    gw = magent.gridworld
    cfg = gw.Config()
    # args.map_size = 12
    # args.n_agents = 3
    # args.more_walls = 0
    # args.more_enemy = 0
    # args.random_num = 1
    # args.mini_map_shape = 6  # battle:20 pursuit:30
    # map_size = args.map_size

    cfg.set({"map_width": map_size, "map_height": map_size})

    # -0.3 -0.15 0
    predator = cfg.register_agent_type("predator", {'width': 1, 'length': 1, 'hp': 1, 'speed': 1, 'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(1), 'attack_penalty': 0})

    prey = cfg.register_agent_type("prey", {'width': 1, 'length': 1, 'hp': 2, 'speed': 0, 'step_recover': 2, 'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)})

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
        self.args.env_name = 'pursuit_easy'
        if self.args.env_name == 'pursuit_hard':
            map_size = 6
            self.args.map_size = map_size
            self.n_agents = 3
            self.args.more_enemy = 0
            self.args.more_walls = 0
            cfg = get_config_pursuit_attack(map_size)
            env = magent.GridWorld(cfg)
        elif args.env_name == 'pursuit_easy':
            map_size = 12
            self.args.map_size = map_size
            self.n_agents = 3
            self.args.more_enemy = 0
            self.args.more_walls = 0
            cfg = get_config_pursuit_attack_ez(map_size)
            env = magent.GridWorld(cfg)
        elif self.args.env_name == 'multi_target':
            map_size = 15  # 80 30
            self.args.map_size = map_size
            self.n_agents = 4  # 6
            self.args.more_enemy = 0
            self.args.more_walls = 0
            cfg = get_config_multi_target(map_size)
            env = magent.GridWorld(cfg)
        elif self.args.env_name == 'double_attack':
            map_size = 10  # 80 30
            self.args.map_size = map_size
            self.n_agents = 4  # 6
            self.args.more_enemy = -3
            self.args.more_walls = 0
            cfg = get_config_double_attack_hard(map_size)
            env = magent.GridWorld(cfg)
        elif self.args.env_name == 'bridge_base':
            map_size = 11  # 80 30
            self.args.map_size = map_size
            self.n_agents = 4  # 6
            self.args.more_enemy = -2
            self.args.more_walls = 0
            cfg = get_config_bridge(map_size)
            env = magent.GridWorld(cfg)
        else:
            raise Exception('env name error')
            # Exception('env name error')
        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env = env
        self.n_actions = env.action_space[0][0]
        self.episode_limit = 100
        handles = env.get_handles()
        feature_dim = env.get_feature_space(handles[0])
        view_dim = env.get_view_space(handles[0])
        real_view_shape = view_dim
        v_dim_total = view_dim[0] * view_dim[1] * view_dim[2]
        obs_shape = (v_dim_total + feature_dim[0], )
        self.args.mini_map_shape = 6
        state_shape = (self.args.mini_map_shape * self.args.mini_map_shape) * 2
        self.view_shape = v_dim_total
        self.act_dim = env.action_space[0][0]
        self.args.fixed_n_actions = env.action_space[1][0]
        self.feature_shape = feature_dim[0]
        self.real_view_shape = real_view_shape
        self.obs_shape = obs_shape[0]
        self.state_shape = state_shape
        self.coding = False
        if not self.coding:
            name = 'cds_' + self.args.env_name + '_01a'
            wandb.init(project="hierarchical MARL", entity="637-muiltagent", name=name, notes='0')
        # self.episode_limit = 100
        # self.view_field = args.map_size
        # self.num_neighbor = args.n_agents - 1
        # self.enemy_feats_dim = 0
        # self.pos_dim = 2
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
        step = 0
        episode_reward = 0
        win_tag = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        # self.env.reset()
        handles = self.env.get_handles()
        # self.env.add_walls(method="random", n=self.n_agents * 2 * self.args.more_walls)
        # self.env.add_agents(handles[0], method="random", n=self.n_agents)
        if self.args.env_name == 'multi_target' or self.args.env_name == 'multi_target_hard':
            # self.env.add_walls(method="random", n=self.n_agents * 2 * self.args.more_walls)
            self.env.add_agents(handles[1],
                                method="custom",
                                n=self.n_agents + self.args.more_enemy,
                                pos=[(1, 1), (1, self.args.map_size - 2), (self.args.map_size - 2, 1), (self.args.map_size - 2, self.args.map_size - 2)])
            self.env.add_agents(handles[0], method="random", n=self.n_agents)
        elif self.args.env_name == 'bridge_base':
            self.generate_map(handles)
        else:
            # self.env.add_walls(method="random", n=self.n_agents * 2 * self.args.more_walls)
            self.env.add_agents(handles[1], method="random", n=self.n_agents + self.args.more_enemy)
            self.env.add_agents(handles[0], method="random", n=self.n_agents)

        while not terminated:
            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < self.n_agents:
                self.env.add_agents(handles[0], method="random", n=self.n_agents - num_agents)
            if fixed_num_agents < (self.n_agents + self.args.more_enemy):
                if 'pursuit' in self.args.env_name:
                    self.env.add_agents(handles[1], method="random", n=(self.n_agents + self.args.more_enemy) - fixed_num_agents)
                elif 'bridge' in self.args.env_name:
                    pos = [(1, 1)]
                    self.env.add_agents(handles[1], method="custom", pos=pos)

            obs_all = self.env.get_observation(handles[0])
            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []
            state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()
            # print(self.n_agents, self.env.get_num(handles[0]))

            for j in range(self.n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                # fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))

            pre_transition_data = {"state": [state], "avail_actions": [np.ones((self.n_agents, self.n_actions))], "obs": [obs]}

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            tmp_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions = []

            for i, action in enumerate(tmp_actions):
                # print(action.type)
                if isinstance(action, np.int64) or isinstance(action, int):
                    actions.append(action.astype(np.int32))
                else:
                    action = action.cpu()
                    actions.append(action.numpy().astype(np.int32))

            acts = [[], []]
            acts[0] = np.array(actions)
            acts[1] = np.array(np.random.randint(0, self.args.fixed_n_actions, size=self.env.get_num(handles[1]), dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            if self.args.env_name == 'bridge_base':
                pos = self.env.get_pos(handles[0])
                reward += self.get_bridge_reward(pos)
            for fr in self.env.get_reward(handles[1]):
                if fr != 0:
                    win_tag += 1
            # fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
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

        obs_all = self.env.get_observation(handles[0])
        fixed_obs_all = self.env.get_observation(handles[1])
        view = obs_all[0]
        feature = obs_all[1]
        fixed_view = fixed_obs_all[0]
        fixed_feature = fixed_obs_all[1]
        obs = []
        fixed_obs = []
        state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()

        for j in range(self.n_agents):
            obs.append(np.concatenate([view[j].flatten(), feature[j]]))
            # fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))

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
        win_tag = win_tag / step / (self.n_agents + self.args.more_enemy)
        episode_reward = round(episode_reward / self.n_agents, 2)

        if test_mode:
            # self.writereward(self.csv_path, np.mean(cur_returns), cur_stats['battle_won'] / cur_stats['n_episodes'], self.t_env)
            # print('=' * 30)
            # print('mean_return', np.mean(cur_returns), 't_env', self.t_env)
            # print(cur_stats)
            # print('=' * 30)
            if not self.coding:
                wandb.log({"magent_recent_return": episode_reward}, step=self.epoch)
                wandb.log({"win rate": win_tag}, step=self.epoch)
                # wandb.log({"win rate": rate}, step=epoch)
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
                reward += 0.1 * (3 - pos[i][1])
        for i in range(2, self.n_agents):
            if pos[i][0] >= 1 and pos[i][0] <= self.args.map_size - 1 and pos[i][1] >= 7 and pos[i][1] <= self.args.map_size - 1:
                reward += 0.1 * (pos[i][1] - 7)
            else:
                reward -= 0
            if pos[i][1] < 7:
                reward += 0.1 * (pos[i][1] - 7)
        return reward
