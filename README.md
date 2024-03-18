# Fuzzy Naive Cartographer

The "model" part of a model-based reinforcement learning algorithm.

## Installation

Pull down the FNC repository.
At the command line: 

```python
git clone git@codeberg.org:brohrer/cartographer.git
```

Install it.

```python
python3 -m pip install -e cartographer
```

## Usage

The FNC class Naive Cartographer is intended to be used in tandem with
a planner (not included) to build a reinforcement learning agent.
At each time step 

1. The FNC makes a prediction
2. The planner uses that prediction to choose an action
3. The FNC registers the action selection

The outline of an agent class looks like this:

```python
from cartographer import NaiveCartographer as Model 

class Agent
    def __init__(self, n_sensors, n_actions):
        model = Model(n_sensors=n_sensors, n_actions=n_actions)
        planner = ...

    def step(self, sensors, reward):
        predictions, rewards, uncertainties = model.predict(sensors, reward)
        actions = planner.choose_actions(predictions, rewards, uncertainties)
        model.update_actions(actions)
        return actions
```


## About

A Fuzzy Naive Cartographer (FNC) builds a
Markov Decision Process-like model of
its world in the form of a set of sequences.
Sequences are of the form of state-action-state as in MDPs.
Here they're also referred to as feature-action-outcome
sequences, to disambiguate the before- and after-state.

FNC builds a value function too.
It creates a set of feature-action pairs and associates a reward
with each. This is analogous to the state-action
value functions of Q-learning.

A model and a value function allow for prediction and planning.
Knowing the current active features and recent actions,
both the reward and the resulting features can be anticipated.

Technically FNC is just the "model" part of a model-based
RL algorithm. It needs to paired with a planner, some element that
will choose an action or goal, to be a complete RL algorithm.
There are some rudimentary planners included with this module
to get you up and running.

There is more detail in the white paper at
[https://brandonrohrer.com/cartographer](https://brandonrohrer.com/cartographer)
