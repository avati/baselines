import os, gym
from time import gmtime, strftime
import numpy
import tensorflow as tf
from baselines import deepq
import csv
from baselines.common import models

from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari

TODO: PROBABLY LINES HAVE BEEN ADDED THT SHOULDNT HAVE BEEN ADDED, AND VICE VERSA
PROBABLY HYPERPARMAS ARE REALLY BAD!
PROBABLY CSV TO SAVE IS MISSING FIELDS, AND VICE VERSA.
NOTE: see how to rewrite line of headers if header disagrees

# Hyperparams
env_name = 'PongNoFrameskip-v4'

TOTAL_TS = int(1e7) # 50000 #100000 # total nb of steps in the training process
MAX_TS = 10000 # 10000 #100000 # max nb of steps after which we stop learning
EXPLORE_TS = 1000. # 1000. #10000 
MAX_NB_XPS = 50 # max nb of xps to run

network='conv_only
buffer_size = 10000
exploration_final_eps = 0.01
print_freq = 10
checkpoint_freq = 1000
lr = 1e-4


# Sanity check
assert EXPLORE_TS / MAX_TS == 0.1


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= MAX_TS
    return is_solved

def main():
    
    # Run the xp MAX_NB_XPS times
    for seed in range(MAX_NB_XPS): 
        tf.reset_default_graph()
        with tf.Session() as session:

            # Get xp_id
            xp_id = strftime("%Y-%m-%d.%H:%M:%S", gmtime())
            print('Experiment: '+xp_id+"_"+env_name)

            dir_to_save = os.path.join('.','save',xp_id)
            if not os.path.exists(dir_to_save):
                os.makedirs(dir_to_save)
            path_to_save = os.path.join(dir_to_save,'pong_model.pkl')

            # Set randomness
            seed = seed
            numpy.random.seed(seed)
            tf.set_random_seed(seed)

            logger.configure()
            env = gym.make(env_name)
            env._max_episode_steps = MAX_TS
            
            logger.configure()
            env = bench.Monitor(env, logger.get_dir())
            env = deepq.wrap_atari_dqn(env)

            act = deepq.learn(
                env,
                network=network,
                convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                hiddens=[256],
                dueling=True,
                seed = seed,
                lr=lr,
                total_timesteps=TOTAL_TS, #100000
                buffer_size=buffer_size,
                exploration_fraction=EXPLORE_TS/MAX_TS,
                exploration_final_eps=exploration_final_eps,
                print_freq=print_freq,
                learning_starts=10000,
                target_network_update_freq=1000,
                gamma=0.99,
                #callback=callback, #--> not for this environment! 
                checkpoint_freq=checkpoint_freq, #10000
                checkpoint_path=dir_to_save,
                param_noise=True
            )            
            
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
    )


            print("Saving .pkl model to: ",path_to_save)
            act.save(path_to_save)
            env.close()

        # Save this run to csv  
        with open(os.path.join('save','results.csv'), 'a', newline='') as csvfile:
            
            

            fieldnames = ['xp_id','env_name','network','seed','lr',\
                          'buffer_size','exploration_fraction',\
                          'exploration_final_eps',\
                          'print_freq','checkpoint_path',\
                         'TOTAL_TS','MAX_TS','EXPLORE_TS','MAX_NB_XPS']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                        'xp_id':xp_id,'env_name':env_name,'network':network,'seed':str(seed),'lr':str(lr),\
                        'buffer_size':buffer_size, 'exploration_fraction':EXPLORE_TS/MAX_TS, \
                        'exploration_final_eps':exploration_final_eps, \
                        'print_freq':print_freq,'checkpoint_path':dir_to_save,\
                        'TOTAL_TS':TOTAL_TS,'MAX_TS':MAX_TS,'EXPLORE_TS':EXPLORE_TS,'MAX_NB_XPS':MAX_NB_XPS,\
                })


if __name__ == '__main__':
    main()

    
    
    