import gym

from baselines import deepq

MAX_TS = 100000
EXPLORE_TS = 10000.

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= MAX_TS
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_TS
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-4,
        total_timesteps=MAX_TS,
        buffer_size=50000,
        exploration_fraction=EXPLORE_TS/MAX_TS,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
