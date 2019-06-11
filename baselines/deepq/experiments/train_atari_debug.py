import os, gym
from time import gmtime, strftime
import numpy
import tensorflow as tf
from baselines import deepq
import csv

# To delete
from baselines.common import models

from baselines import logger
from baselines.common.atari_wrappers import make_atari
from baselines import bench


# Hyperparams
env_name = 'EnduroNoFrameskip-v4'
alpha = 0.
dueling=False
prioritized_replay=False

# Xp params
TOTAL_TS = 2500000 #500000 #int(1e6) # total nb of steps in the training process
MAX_NB_XPS = 10 # max nb of xps to run
MAX_TS = 10000 # max nb of steps after which we stop learning
EXPLORE_TS = 1000. 
# Sanity check
assert EXPLORE_TS / MAX_TS == 0.1

# MODEL PARAMS

# MLP: for 'CartPole-v0'
# dafault 'mlp' is the same as: mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False)
#network= 'models.mlp( num_layers=3, num_hidden=128)' #'mlp','conv_only # network is shared: should be deep. and conv layers

# CONV: for Pong
network='conv_only'
convs= [(32, 8, 4), (64, 4, 2), (64, 3, 1)] 

if not network in ['mlp','conv_only']:
    network_ = network
    network = eval(network)
else: 
    network_ = network
print('network_',network_)
print('network',network)

hiddens= [] # it is the final non-shared layers: should be shallow

buffer_size = 10000
exploration_final_eps = 0.01
print_freq = 10
checkpoint_freq = 5000
lr = 1e-4

"""
# Not used for Pong
def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= MAX_TS
    return is_solved
"""

def main():
    
    # Run the xp MAX_NB_XPS times
    for seed in range(MAX_NB_XPS): 
        tf.reset_default_graph()
        with tf.Session() as session:

            # Get xp_id
            xp_id = strftime("%Y-%m-%d.%H:%M:%S", gmtime()) +"_env="+env_name + "_alpha=" + str(alpha) \
                    + "_dueling=" + str(dueling) + "_prioritized_replay=" + str(prioritized_replay)
            print('Experiment: '+xp_id)

            dir_to_save = os.path.join('.','save',xp_id)
            if not os.path.exists(dir_to_save):
                os.makedirs(dir_to_save)
            path_to_save = os.path.join(dir_to_save,'atari_model.pkl')

            # Set randomness
            seed = seed
            numpy.random.seed(seed)
            tf.set_random_seed(seed)

            logger.configure(dir_to_save) #checkpoint_path 
            env = make_atari(env_name)
            env = bench.Monitor(env, logger.get_dir())
            env = deepq.wrap_atari_dqn(env)
            
            act = deepq.learn(
                env,
                network=network, #models.conv_only(num_hidden=64, num_layers=1), 
                convs=convs,
                hiddens=hiddens,
                dueling=dueling,
                prioritized_replay=prioritized_replay,
                seed = seed,
                lr=lr,
                total_timesteps=TOTAL_TS, #100000
                buffer_size=buffer_size,
                exploration_fraction=EXPLORE_TS/MAX_TS,
                exploration_final_eps=exploration_final_eps,
                print_freq=print_freq,
                #callback=callback, #not in pong!
                checkpoint_freq=checkpoint_freq, #10000
                checkpoint_path=dir_to_save,
                alpha=alpha,
                train_freq=4,
                learning_starts=10000,
                target_network_update_freq=1000,
                gamma=0.99,
            )
            
            print("Saving .pkl model to: ",path_to_save)
            act.save(path_to_save)
            
            env.close()

        # Save this run to csv  
        with open(os.path.join('save','results.csv'), 'a', newline='') as csvfile:
            
            fieldnames = ['xp_id','env_name','network', 'hiddens','convs',\
                          'dueling', 'prioritized_replay', \
                          'seed','lr','buffer_size', 'exploration_fraction','exploration_final_eps',\
                          'print_freq','checkpoint_path',\
                         'TOTAL_TS','EXPLORE_TS','MAX_NB_XPS','alpha']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                        'xp_id':xp_id,'env_name':env_name,'network':network_, 'hiddens':str(hiddens),'convs':str(convs), \
                        'dueling':dueling,'prioritized_replay':prioritized_replay,
                        'seed':str(seed),'lr':str(lr),\
                        'buffer_size':buffer_size, 'exploration_fraction':EXPLORE_TS/MAX_TS, \
                        'exploration_final_eps':exploration_final_eps, \
                        'print_freq':print_freq,'checkpoint_path':dir_to_save,\
                        'TOTAL_TS':TOTAL_TS,'EXPLORE_TS':EXPLORE_TS,'MAX_NB_XPS':MAX_NB_XPS,\
                        'alpha':alpha,\
                })


if __name__ == '__main__':
    main()

    
