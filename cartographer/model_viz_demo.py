import time
import numpy as np
import cartographer.model_viz as model_viz


def main():
    viz = model_viz.Visualization()

    n_sensors = 3
    n_actions = 4

    def step():
        rewards = np.random.sample(n_actions) * 2 - 1
        predictions = np.random.sample((n_actions, n_sensors))
        uncertainties = np.random.sample(n_actions)

        viz.update(predictions, rewards, uncertainties)

    for i in range(100):
        print(i)
        time.sleep(1)
        step()


if __name__ == "__main__":
    main()
