import argparse
import datetime
import numpy as np
import itertools
import torch
from sac import SAC
from replay_memory import ReplayMemory

from custom_env import CustomEnv


class Workspace(object):
    def __init__(self, sac_hyperparams, rlb_env_config):
        self.sac_hyparams = sac_hyperparams
        self.rlb_env_config = rlb_env_config
        
        # Environment
        self.env = CustomEnv(rlb_env_config)
        obs, task_obs, state_obs = self.env.reset()
        if rlb_env_config['task'] in ['ReachTarget', 'PushButton']:
            self.state_dim = 8+3 #ReachTarget:8+3, PushButton:8+3, CloseMicrowave:8+11
        elif rlb_env_config['task'] in ['CloseMicrowave']:
            self.state_dim = 8+11
        else:
            self.state_dim = 11
        action_size = self.env.env.action_size
        action_high = np.ones(action_size, dtype=np.float32)
        action_low = np.ones(action_size, dtype=np.float32)*(-1)
        action_low[-1] = 0.0
        self.action_dim = argparse.Namespace(**{'high': action_high, 'low': action_low, 'shape': (action_size,)})
        
        torch.manual_seed(sac_hyperparams.seed)
        np.random.seed(sac_hyperparams.seed)
        
        # Agent
        self.agent = SAC(self.state_dim, self.action_dim, args=sac_hyperparams)
        
        # Memory
        self.agent_memory = ReplayMemory(sac_hyperparams.replay_size, sac_hyperparams.seed)
        
        # Training Loop
        self.total_numsteps = 0
        self.updates = 0
        
        self.rank_count = 0

        self.start_steps_1 = 150000
        
        
    def evaluate(self, i_episode):
        print("----------------------------------------")
        for _  in range(i_episode):
            obs, task_obs, state_obs = self.env.reset()
            state = np.concatenate([state_obs, task_obs], axis=-1)
            state[state==None] = 0.0
            state = state.astype(np.float32)

            episode_reward = 0
            done = False
            episode_steps = 0
            while not done:
                action = self.agent.select_action(state, evaluate=True)

                next_obs, next_task_obs, next_state_obs, reward, done = self.env.step(action) # Step
                next_state = np.concatenate([next_state_obs, next_task_obs], axis=-1)
                next_state[next_state == None] = 0.0
                next_state = next_state.astype(np.float32)
                episode_reward += reward
                state = next_state
                episode_steps += 1
                if episode_steps % self.rlb_env_config['episode_len'] == 0:
                    done = True
            print("Reward: {}".format(round(episode_reward, 2)))
        print("----------------------------------------")
        
    def train(self):

        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            obs, task_obs, state_obs = self.env.reset()
            state = np.concatenate([state_obs, task_obs], axis=-1)
            state[state==None] = 0.0
            state = state.astype(np.float32)

        
            while not done:
                if self.sac_hyparams.start_steps > self.total_numsteps:
                    action = self.env.randn_action()  # Sample random action
                else:
                    action = self.agent.select_action(state)  # Sample action from policy
        
                if len(self.agent_memory) > self.sac_hyparams.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.sac_hyparams.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.agent_memory, self.sac_hyparams.batch_size, self.updates)
                        self.updates += 1
        
                next_obs, next_task_obs, next_state_obs, reward, done = self.env.step(action) # Step
                next_state = np.concatenate([next_state_obs, next_task_obs], axis=-1)
                next_state[next_state == None] = 0.0
                next_state = next_state.astype(np.float32)

                episode_steps += 1
                self.total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == 200 else float(not done)

                self.agent_memory.push(state, action, reward, next_state, mask) # push data to agent memory

                if episode_steps % self.rlb_env_config['episode_len'] == 0:
                    done = True
        
                state = next_state
        
            if i_episode > self.sac_hyparams.max_episodes:
                break

            print("E{}, t numsteps: {}, e steps: {}, reward: {}".format(i_episode, self.total_numsteps, episode_steps,
                                                                                                            round(episode_reward, 2)))
        
            if i_episode % self.sac_hyparams.eval_per_episode == 0 and self.sac_hyparams.eval is True:
                self.evaluate(3)
                
        # self.env.close()





if __name__ == '__main__':
    sac_hyperparams = {
        'policy': "Gaussian",  #Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval': True,  #Evaluates a policy a policy every 10 episode (default: True)
        'eval_per_episode': 50,  #evaluate policy per episode
        'render': False,  #Render when evaluate policy
        'test_episodes': 3,
        'gamma': 0.99,  
        'tau': 0.005,  #target smoothing coefficient(τ) (default: 0.005)
        'lr': 0.0003,#default=0.0003 / 0.00005
        'alpha': 0.2,  #Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)
        'automatic_entropy_tuning': True,  #Automaically adjust α (default: False)
        'seed': 123456,  #random seed (default: 123456)
        'batch_size': 256,
        'max_steps': 50000,  #maximum number of steps (default: 1000000)
        'max_episodes': 8000,  #maximum number of episodes (default: 3000)
        'hidden_size': 256,
        'updates_per_step': 1,  #model updates per simulator step (default: 1)
        'start_steps': 10000,  #Steps sampling random actions (default: 10000 , 200000)
        'target_update_interval': 1,  #Value target update per no. of updates per step (default: 1)
        'replay_size': 1000000,  #size of replay buffer (default: 10000000)
        'cuda': True
        }

    rlb_env_config = {
        'task': "PushButton",  #
        'static_env': False,  #
        'headless_env': False,  #
        'learn_reward_frequency': 100,  #
        'episode_len': 150,  #
        'obs_type': "LowDimension"  # LowDimension WristCameraRGB
    }

    
    sac_hyperparams = argparse.Namespace(**sac_hyperparams)
    
    
    Workspace = Workspace(sac_hyperparams, rlb_env_config)
    
    
    Workspace.train()
           