import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np




class envWrapperNav(gym.Env):
    def __init__(self, RDDLEnv, max_episode_length=100):
        super(envWrapperNav, self).__init__()
        self.RDDLEnv = RDDLEnv
        
        self.action_list = sorted(self.RDDLEnv.action_space.keys())
        self.observation_list = sorted(self.RDDLEnv.observation_space.keys())

        self.action_space = Discrete(len(self.action_list))
        self.observation_space = self.new_observation_space(self.RDDLEnv)

        self.state = RDDLEnv.reset()

        self.max_episode_length = max_episode_length
        self.current_step = 0
    
    def convert_dict_numpy(self, int_dict):
        np_dict = {}
        for k,v in int_dict.items():
            np_dict[k] = np.array([v], dtype=np.float32)
        return np_dict


    def action_vec2dict(self, action_num):
        action_dict = {}
        for idx, action_str in enumerate(self.action_list):
            v = 1 if idx == action_num else 0
            action_dict[action_str] = v
        return action_dict

    def action_dict2vec(self, action_dict):
        action_vec = np.zeros(len(self.action_list))
        for idx, action in enumerate(self.action_list):
            action_vec[idx] = action_dict[action]
        return action_vec
    
    def new_observation_space(self, RDDLEnv):
        new_observation_space = Dict()
        for k,v in RDDLEnv.observation_space.items():
            low = v.low
            high = v.high
            shape = v.shape
            new_observation_space[k] = Box(low=low,high=high,shape=v.shape,dtype=np.float32)
        return new_observation_space

    def reset(self, seed=None):
        state = self.convert_dict_numpy(self.RDDLEnv.reset(seed))

        self.current_step = 0

        return (state, {})
    
    def step(self, action):
        action_dict = self.action_vec2dict(action)
        next_state, reward, done, info = self.RDDLEnv.step(action_dict)

        self.current_step += 1

        if self.current_step >= self.max_episode_length:
            done = True
        else:
            done = False
        
        
        self.state = self.convert_dict_numpy(next_state)   

        return self.state, reward, done, False, info  # Modify as needed

