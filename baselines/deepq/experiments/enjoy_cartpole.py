import os, gym

from baselines import deepq

MAX_TS = 1000000

def main():
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_TS

    
    # Define which xp to load
    xp_id = '2019-06-08.07:15:01'
    dir_to_save = os.path.join('.','save',xp_id)
    if not os.path.exists(dir_to_save):
        raise ValueError('Directory does not exist: ',dir_to_save)
    print('Experiment: '+xp_id)
    
    path_to_save = os.path.join(dir_to_save,'cartpole_model.pkl')
    checkpoint_path=dir_to_save

    # Find out which one is correct
    load_path_1 = path_to_save # seems to work
    #load_path_2 = checkpoint_path # bug
    load_path_3 = os.path.join(checkpoint_path, "model") # seems to work

    print('load_path_1', load_path_1)
    #print('load_path_2', load_path_2)
    print('load_path_3', load_path_3)
    
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path=load_path_1)
    
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:

            env.render()
            
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()