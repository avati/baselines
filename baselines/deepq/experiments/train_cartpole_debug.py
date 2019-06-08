import os, gym
from time import gmtime, strftime
import numpy
import tensorflow as tf
from baselines import deepq
import csv
from baselines import logger


# Hyperparams
env_name = 'CartPole-v0'

# TO MODIFY
TOTAL_TS = 1000 #500000 # total nb of steps in the training process
MAX_NB_XPS = 10 # max nb of xps to run

MAX_TS = 10000 # max nb of steps after which we stop learning
EXPLORE_TS = 1000. 

network='mlp'
buffer_size = 50000
exploration_final_eps = 0.02
print_freq = 10
checkpoint_freq = 10000
lr = 1e-4

alpha = 0.1

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
            xp_id = strftime("%Y-%m-%d.%H:%M:%S", gmtime()) +"_env="+env_name + "_alpha=" + str(alpha)
            print('Experiment: '+xp_id)

            dir_to_save = os.path.join('.','save',xp_id)
            if not os.path.exists(dir_to_save):
                os.makedirs(dir_to_save)
            path_to_save = os.path.join(dir_to_save,'cartpole_model.pkl')

            # Set randomness
            seed = seed
            numpy.random.seed(seed)
            tf.set_random_seed(seed)

            logger.configure(dir_to_save) #checkpoint_path
            
            env = gym.make(env_name)
            env._max_episode_steps = MAX_TS
            
            act = deepq.learn(
                env,
                network=network,
                seed = seed,
                lr=lr,
                total_timesteps=TOTAL_TS, #100000
                buffer_size=buffer_size,
                exploration_fraction=EXPLORE_TS/MAX_TS,
                exploration_final_eps=exploration_final_eps,
                print_freq=print_freq,
                callback=callback,
                checkpoint_freq=checkpoint_freq, #10000
                checkpoint_path=dir_to_save,
                alpha=alpha
            )

            print("Saving .pkl model to: ",path_to_save)
            act.save(path_to_save)
            
            env.close()

        # Save this run to csv  
        with open(os.path.join('save','results.csv'), 'a', newline='') as csvfile:
            
            fieldnames = ['xp_id','env_name','network','seed','lr','buffer_size','exploration_fraction','exploration_final_eps',\
                          'print_freq','checkpoint_path',\
                         'TOTAL_TS','MAX_TS','EXPLORE_TS','MAX_NB_XPS','alpha']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                        'xp_id':xp_id,'env_name':env_name,'network':str(network),'seed':str(seed),'lr':str(lr),\
                        'buffer_size':buffer_size, 'exploration_fraction':EXPLORE_TS/MAX_TS, \
                        'exploration_final_eps':exploration_final_eps, \
                        'print_freq':print_freq,'checkpoint_path':dir_to_save,\
                        'TOTAL_TS':TOTAL_TS,'MAX_TS':MAX_TS,'EXPLORE_TS':EXPLORE_TS,'MAX_NB_XPS':MAX_NB_XPS,\
                        'alpha':alpha,\
                })


if __name__ == '__main__':
    main()

    
