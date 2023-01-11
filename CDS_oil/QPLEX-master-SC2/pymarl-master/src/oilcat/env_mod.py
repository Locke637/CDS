"""
# @Time    : 2021/7/2 5:22 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env.py
"""

import numpy as np
from mappo.envs.oil_env import OilCat, configs


class Env(object):
    """
    # 环境中的智能体
    """

    def __init__(self, i):
        # self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个
        # self.obs_dim = 14  # 设置智能体的观测纬度
        # self.action_dim = 5  # 设置智能体的动作纬度，这里假定为一个五个纬度的
        self.oilcat = OilCat(configs)
        oilcat = self.oilcat
        self.agent_num = oilcat.agent_num
        self.obs_dim = oilcat.obs_dim[0]
        self.action_dim = oilcat.act_dim[0]

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        return self.oilcat.reset()

    def step(self, actions):
        # print(actions)
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """
        # sub_agent_obs = []
        # sub_agent_reward = []
        # sub_agent_done = []
        # sub_agent_info = []

        assert len(actions) == self.agent_num

        # print(type(actions))

        actions_number = np.argmax(actions, axis=-1)
        (obs, rewards, done, info) = self.oilcat.step(actions_number)
        # print(self.oilcat.oil_visited)
        if done:
            obs = self.oilcat.reset()
            # print(info)
        # for i in range(self.agent_num):
        #     sub_agent_obs.append(np.random.random(size=(self.obs_dim,)))
        #     sub_agent_reward.append([np.random.rand()])
        #     sub_agent_done.append(False)
        #     sub_agent_info.append({})

        return [
            obs,
            np.expand_dims(rewards, axis=-1),
            [done] * self.agent_num,
            [info] * self.agent_num,
        ]
