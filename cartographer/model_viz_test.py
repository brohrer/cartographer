import time
import numpy as np
import matplotlib.pyplot as plt
import model_viz

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
