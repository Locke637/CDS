import argparse
import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

DIR_TO_VEC = {
    "stop": np.array([0, 0]),
    "left": np.array([-1, 0]),
    "right": np.array([1, 0]),
    "up": np.array([0, 1]),
    "down": np.array([0, -1]),
}

ACT_TO_DIR = {0: "stop", 1: "left", 2: "right", 3: "up", 4: "down"}


def vision_to_vet(vision):
    point_list = []
    for i in range(vision + 1):
        for j in range(vision + 1):
            kx = [1]
            ky = [1]
            if i != 0:
                kx.append(-1)
            if j != 0:
                ky.append(-1)
            for x in kx:
                for y in ky:
                    point_list.append(np.array([i * x, j * y]))

    return point_list


class OilCat:
    def __init__(self, configs):
        self.configs = configs

        self.L = configs["L"]
        self.periodic_boundary = configs["periodic_boundary"]
        self.reward_style = configs["reward_type"]

        self.oil_num = configs["oil_num"]
        self.oil_values = configs["oil_values"]
        self.oil_locations = configs["oil_locations"]
        self.oil_reaching = configs["oil_reaching"]
        self.oil_length = configs["oil_length"]
        self.grid_values = np.zeros((self.L, self.L))
        self.grid_values_ch = np.zeros((self.oil_num, self.L, self.L))
        self.discount = configs["discount"]
        self.init_grid_values()

        self.agent_num = configs["agent_num"]
        self.agent_vision = configs["agent_vision"]
        self.agent_start = configs["agent_start"]

        self.max_step = configs["max_step"]

        self.cur_step = 0
        self.oil_visited = [0 for _ in range(self.oil_num)]
        self.temp_grid_values = np.zeros((self.L, self.L))
        self.visited_agent = []

        self.obs_dim, self.act_dim = self.env_info()

        self.snap_shot = {
            "step": -1,
            "agent_current_locations": [],
            "agent_action_sequences": [],
            "agent_current_rewards": np.zeros(self.agent_num),
            "agent_accumulate_rewards": np.zeros(self.agent_num),
            "agent_current_obs": [],
        }

    def env_info(self):
        obs_dim = []
        for i in range(self.agent_num):
            obs_dim.append(2 + (self.oil_num + 1) * (2 * self.agent_vision[i] + 1)**2)
        act_dim = [len(ACT_TO_DIR) for _ in range(self.agent_num)]

        return obs_dim, act_dim

    def init_grid_values(self):
        # print()
        point_list = vision_to_vet(self.oil_length)
        # print(point_list)
        for oil in range(self.oil_num):
            oil_point = np.array(self.oil_locations[oil])
            # print(len(oil_point))
            for point in point_list:
                distance = np.sqrt(np.sum(np.square(point)))
                distance = self.discount * distance
                # print(distance)
                tar_point = oil_point + point
                # print(tar_point)
                if self.periodic_boundary:
                    tar_point = tar_point % np.array([self.L, self.L])
                else:
                    if (tar_point[0] < 0 or tar_point[0] >= self.L or tar_point[1] < 0 or tar_point[1] >= self.L):
                        continue
                tar_x = int(tar_point[0])
                tar_y = int(tar_point[1])
                # print(tar_x)
                self.grid_values[tar_x][tar_y] = np.maximum(
                    self.grid_values[tar_x][tar_y],
                    self.oil_values[oil] / (distance + 1),
                )
                self.grid_values_ch[oil][tar_x][tar_y] = self.oil_values[oil] / (distance + 1)
        # self.temp_grid_values = deepcopy(self.grid_values)

    def refine_grid_values(self):
        # print("refine")
        self.temp_grid_values = np.zeros((self.L, self.L))

        point_list = vision_to_vet(self.oil_length)
        # print(point_list)
        for oil in range(self.oil_num):
            if self.oil_visited[oil] == 1:
                # print("visited")
                continue
            oil_point = np.array(self.oil_locations[oil])
            # print(len(oil_point))
            for point in point_list:
                distance = np.sqrt(np.sum(np.square(point)))
                distance = self.discount * distance
                # print(distance)
                tar_point = oil_point + point
                # print(tar_point)
                if self.periodic_boundary:
                    tar_point = tar_point % np.array([self.L, self.L])
                else:
                    if (tar_point[0] < 0 or tar_point[0] >= self.L or tar_point[1] < 0 or tar_point[1] >= self.L):
                        continue
                tar_x = int(tar_point[0])
                tar_y = int(tar_point[1])
                # print(tar_x)
                self.temp_grid_values[tar_x][tar_y] = np.maximum(
                    self.grid_values[tar_x][tar_y],
                    self.oil_values[oil] / (distance + 1),
                )
                self.grid_values_ch[oil][tar_x][tar_y] = self.oil_values[oil] / (distance + 1)

    def reset(self):
        self.cur_step = 0
        self.temp_grid_values = deepcopy(self.grid_values)
        self.oil_visited = [0 for _ in range(self.oil_num)]

        self.snap_shot["step"] = 0
        self.snap_shot["agent_previous_locations"] = np.zeros((self.agent_num, 2))
        self.snap_shot["agent_current_locations"] = np.zeros((self.agent_num, 2))
        self.snap_shot["agent_action_sequences"] = []
        self.snap_shot["agent_current_obs"] = []
        for i in range(self.agent_num):
            self.snap_shot["agent_current_locations"][i] = np.array(self.agent_start[i]) % np.array([self.L, self.L])
            self.snap_shot["agent_action_sequences"].append([])
            self.snap_shot["agent_current_obs"].append([])
        self.snap_shot["agent_current_rewards"] = np.zeros(self.agent_num)
        self.snap_shot["agent_accumulate_rewards"] = np.zeros(self.agent_num)
        self.visited_agent = []

        return self.get_obs()

    # def get_obs(self):
    #     # print()
    #     obs = []
    #     for i in range(self.agent_num):
    #         current_location = self.snap_shot["agent_current_locations"][i]
    #         point_list = vision_to_vet(self.agent_vision[i])
    #         # print(len(point_list))
    #         point_values = np.zeros([len(point_list)])
    #         point_taken = np.zeros([len(point_list)])
    #         for j, point in enumerate(point_list):
    #             temp_location = (current_location + point) % np.array([self.L, self.L])
    #             point_values[j] = self.grid_values[int(temp_location[0])][int(temp_location[1])]

    #             for k in range(agent_num):
    #                 if (temp_location == self.snap_shot["agent_current_locations"][k]).all():
    #                     point_taken[j] = 1

    #         obs.append(np.concatenate([current_location, point_values, point_taken], axis=-1))
    #     return obs

    def get_obs(self):
        # print()
        obs = []
        for i in range(self.agent_num):
            current_location = self.snap_shot["agent_current_locations"][i]
            point_list = vision_to_vet(self.agent_vision[i])
            # print(len(point_list))
            point_values = np.zeros([len(point_list) * self.oil_num])
            point_taken = np.zeros([len(point_list)])
            for j, point in enumerate(point_list):
                temp_location = (current_location + point) % np.array([self.L, self.L])
                for oil_id in range(self.oil_num):
                    point_values[j + oil_id * len(point_list)] = self.grid_values_ch[oil_id][int(temp_location[0])][int(temp_location[1])]

                for k in range(self.agent_num):
                    if (temp_location == self.snap_shot["agent_current_locations"][k]).all():
                        point_taken[j] = 1

            obs.append(np.concatenate([current_location, point_values, point_taken], axis=-1))
        return obs

    def get_reward(self):
        oil_agent_taken = np.zeros([self.agent_num, self.oil_num])

        for i in range(self.agent_num):
            if i not in self.visited_agent:
                current_location = self.snap_shot["agent_current_locations"][i]
                for j in range(self.oil_num):
                    if (current_location == self.oil_locations[j]).all():
                        oil_agent_taken[i][j] = 1
                        self.visited_agent.append(i)
        reward = 0
        flag = False
        # print(self.oil_visited)
        for j in range(self.oil_num):
            taken = np.sum(oil_agent_taken[:, j])
            if taken >= self.oil_reaching[j] and not self.oil_visited[j]:
                reward += self.oil_values[j]
                self.oil_visited[j] = 1
                flag = True
        if flag:
            # print("refine")
            self.refine_grid_values()
        # # print(np.array(self.oil_visited).all())
        # if np.array(self.oil_visited).all():
        #     reward += sum(self.oil_values)
        return np.array([reward for _ in range(self.agent_num)])

    def get_avail_actions(self, index):
        # avail_actions = []
        if index in self.visited_agent:
            avail_action = [1, 0, 0, 0, 0]
        else:
            avail_action = [1, 1, 1, 1, 1]
        return avail_action

    def get_state(self):
        state = np.concatenate([np.array(self.snap_shot['agent_current_locations']).flatten(), np.array(self.oil_locations).flatten()])
        return state

    def step(self, actions):
        assert len(actions) == self.agent_num

        self.cur_step += 1
        self.snap_shot["step"] = self.cur_step

        self.snap_shot["agent_previous_locations"] = self.snap_shot["agent_current_locations"]
        # current_locations = deepcopy(previous_locations)
        for i in range(self.agent_num):
            self.snap_shot["agent_current_locations"][i] = (self.snap_shot["agent_current_locations"][i] + DIR_TO_VEC[ACT_TO_DIR[actions[i]]]) % np.array([self.L, self.L])
        obs = self.get_obs()
        # print(len(obs[0]))
        rewards = self.get_reward()
        # print(rewards)
        done = False

        if self.cur_step >= self.max_step or (0 not in self.oil_visited):
            done = True

        info = {
            "max_return": sum(self.oil_values),
            "oil_values": self.oil_values,
            "visited": self.oil_visited,
        }
        return obs, rewards, done, info

        # print(previous_locations)
        # print(current_locations)

    # def grid_print(self):
    #     for i in range(len(self.grid_values)):
    #         for j in range(len(self.grid_values[i])):
    #             print("%.{}f ".format(2) % self.grid_values[i][j], end="")
    #         print()
    def render(self, save=False):
        plt.figure(figsize=(8, 8))
        # self.temp_grid_values[20][10] = -3000
        grid_plot = sns.heatmap(
            self.temp_grid_values,
            cmap="magma_r",
            linewidths=0.02,
            cbar=True,
            vmin=0,
            vmax=self.configs["max_oil_value"],
        )
        # grid_plot.invert_yaxis()
        # grid_plot.invert_xaxis()
        grid_plot.set(xticklabels=[])
        grid_plot.set(yticklabels=[])
        plt.axis("off")

        # plt.scatter(y=[20], x=[10], marker='+', c='r')
        s = [20 * 2**(self.oil_reaching[i] + 1) for i in range(self.oil_num)]
        plt.scatter(
            y=self.oil_locations[:, 0] + 0.5,
            x=self.oil_locations[:, 1] + 0.5,
            marker="*",
            c="r",
            s=s,
        )

        # print(self.snap_shot["agent_current_locations"].shape)
        plt.scatter(
            y=self.snap_shot["agent_current_locations"][:, 0] + 0.5,
            x=self.snap_shot["agent_current_locations"][:, 1] + 0.5,
            marker="x",
            c="b",
        )
        if save:
            plt.savefig(
                "grid.pdf",
                bbox_inches="tight",
                pad_inches=0.0,
            )
        plt.show()

    def get_neighbor_position(self):
        neighbor_position = {}
        neighbor_list = {}
        for i in range(self.agent_num):
            neighbor_list[i] = []
            neighbor_position[i] = []
            for j in range(self.agent_num):
                if i != j:
                    neighbor_list[i].append(j)
                    neighbor_position[i].append(np.array(self.snap_shot["agent_current_locations"][j]) - np.array(self.snap_shot["agent_current_locations"][i]))
        return neighbor_list, neighbor_position

    def close(self):
        pass


# parser = argparse.ArgumentParser()
# parser.add_argument("--L", type=int, default=100, help="length of the grid")
# parser.add_argument("--oil_num", type=int, default=10, help="number of oilfields")
#
# args = parser.parse_args()

# def merge_configs(args):
#     print()
#     return configs


def generate_random_n_values(n):
    temp = np.random.rand(n)
    return temp / np.sum(temp)


# L = 10  #30 20
# oil_num = 2  #10
# max_value = 100
# # oil_locations = []
# # oil_values = []
# # for i in range(oil_num):
# oil_locations = np.random.randint(0, L, (oil_num, 2))
# # oil_locations = np.array([[1, 1], [L - 2, L - 2]])
# # oil_values = generate_random_n_values(oil_num) * max_value
# oil_values = [70, 30]
# # oil_values = np.random.rand(oil_num) * max_value

# agent_num = 4  #20
# # agent_start_point = []
# agent_start_point = np.random.randint(0, L, (agent_num, 2))

# # print(oil_locations)
# configs = {
#     "L": L,
#     "oil_num": oil_num,
#     "oil_values": oil_values,
#     "oil_locations": oil_locations,
#     "oil_length": 3,
#     "agent_num": agent_num,
#     "agent_start": agent_start_point,
#     "agent_vision": [1 for _ in range(agent_num)],
#     "periodic_boundary": True,
#     "reward_type": "step",
#     "oil_reaching": [2, 2],  #[3, 2, 2, 3, 2, 2, 1, 1, 2, 1]
#     "max_step": 40,  #30
#     "max_oil_value": max_value,
# }

# oc = OilCat(configs=configs)
#
# oc.init_grid_values()
# oc.reset()
# oc.render(save=True)
# oc.grid_print()
# for _ in range(configs["max_step"]):
#     oc.step([np.random.randint(len(ACT_TO_DIR)) for i in range(agent_num)])
# oc.render(save=False)

# grid_plot = sns.heatmap(oc.grid_values, cmap="magma", linewidths=0.02)
# grid_plot.set(xticklabels=[])
# grid_plot.set(yticklabels=[])
# plt.axis("off")
# ,
# plt.scatter(x=oc.snap_shot["agent_current_locations"][:, 0]+0.5,
#             y=oc.snap_shot["agent_current_locations"][:, 1]+0.5, marker='x')
# plt.show()
