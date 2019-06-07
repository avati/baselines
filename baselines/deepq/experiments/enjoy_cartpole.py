import gym

from baselines import deepq
MAX_TS = 1000000

def main():
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_TS
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="cartpole_model.pkl")

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
