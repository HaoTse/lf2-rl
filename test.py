import pandas as pd
import numpy as np
import argparse

from config import MEMORY_CAPACITY, TEST_EPISODE_NUM, BATCH_SIZE, E_GREEDY, MAX_STEP
from brain import DQN

# Add import path and import the lf2gym
import os, sys
sys.path.append(os.path.abspath('lf2gym'))
# Import lf2gym
import lf2gym

def newReward(obsesrvation, obsesrvation_):
    return abs(obsesrvation_[0] - (-0.5))

def transOber(observation):
    observation = np.transpose(observation, (2, 1, 0))
    observation = np.transpose(observation, (0, 2, 1))
    return observation

def test(mothod, model_path):
    #load model
    RL.load_model(model_path)

    steps, rewards = [], []
    for episode in range(TEST_EPISODE_NUM):
        # initial
        observation = env.reset()
        observation = transOber(observation)

        iter_cnt, total_reward = 0, 0
        while True:
            iter_cnt += 1

            # fresh env
            env.render()

            if method == 'DQN':
                # RL choose action based on observation
                action = RL.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward, done, _ = env.step(action)
                observation_ = transOber(observation_)
                # reward = newReward(observation, observation_)

            # accumulate reward
            total_reward += reward
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                total_reward = round(total_reward, 2)
                steps.append(iter_cnt)
                rewards.append(total_reward)
                print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
                break

    # end of game
    print('--------------------------------')
    print('After {} episode,\nthe average step is {},\nthe average reward is {}.'.format(TEST_EPISODE_NUM, sum(steps)/len(steps), sum(rewards)/len(rewards)))
    env.close()

if __name__ == "__main__":

    # argument
    parse = argparse.ArgumentParser()
    parse.add_argument('-m', '--method',
                        default='DQN',
                        help='Choose which rl algorithm used (DQN)')
    parse.add_argument('-t', '--test',
                        default='model/DQN/eval_0.01_{}_{}.pkl'.format(E_GREEDY, BATCH_SIZE),
                        help='The test model path')
    args = parse.parse_args()

    # game setup
    AGENT = 'Davis'
    OPPOENENT = 'Dennis'

    # env setup
    env = lf2gym.make(startServer=True, wrap='skip4', driverType=lf2gym.WebDriver.Chrome, 
        characters=[lf2gym.Character[AGENT], lf2gym.Character[OPPOENENT]], 
        difficulty=lf2gym.Difficulty.Crusher, debug=True)
    
    options = env.get_reset_options()
    print('Original reset options: %s' % options)
    options['hp_full'] = 100
    options['mp_start'] = 250
    print('Custom reset options: %s' % options)

    # algorithm setup
    method = args.method
    if method == 'DQN':
        print("Use DQN...")
        print('--------------------------------')
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
        RL = DQN(action_n=env.action_space.n, state_n=env.observation_space.n, env_shape=env_shape)
    else:
        print("Error method! Use DQN instead.")
        print('--------------------------------')
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
        RL = DQN(action_n=env.action_space.n, state_n=env.observation_space.n, env_shape=env_shape)

    test(method, args.test)