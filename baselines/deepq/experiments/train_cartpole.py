import os, gym
from time import gmtime, strftime

import numpy
import tensorflow

from baselines import deepq

TOTAL_TS = 50000 #100000
MAX_TS = 10000 #100000
EXPLORE_TS = 1000. #10000

assert EXPLORE_TS / MAX_TS == 0.1

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= MAX_TS
    return is_solved

def main():
    # Get xp_id
    xp_id = strftime("%Y-%m-%d.%H:%M:%S", gmtime())
    print('Experiment: '+xp_id+"_CartPole-v0")

    dir_to_save = os.path.join('.','save',xp_id)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    path_to_save = os.path.join(dir_to_save,'cartpole_model.pkl')

    # Set randomness
    seed = 1
    numpy.random.seed(seed)
    tensorflow.set_random_seed(seed)

    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_TS
    act = deepq.learn(
        env,
        network='mlp',
        seed = seed,
        lr=1e-3,
        total_timesteps=TOTAL_TS, #100000
        buffer_size=50000,
        exploration_fraction=EXPLORE_TS/MAX_TS,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        checkpoint_freq=1000, #10000
        checkpoint_path=dir_to_save,
        
    )
    
    print("Saving .pkl model to: ",path_to_save)
    act.save(path_to_save)


if __name__ == '__main__':
    main()

    
