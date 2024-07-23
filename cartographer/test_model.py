import numpy as np
from cartographer.model import NaiveCartographer as Model


def test_initalization():
    n_sensors = 3
    n_actions = 4
    n_rewards = 2
    model = Model(n_sensors=n_sensors, n_actions=n_actions, n_rewards=n_rewards)
    assert model.n_features == n_sensors
    assert model.n_outcomes == n_sensors
    assert model.n_actions == n_actions + 2
    assert model.n_rewards == n_rewards


def test_updates():
    n_sensors = 3
    n_actions = 4
    n_rewards = 2
    model = Model(n_sensors=n_sensors, n_actions=n_actions, n_rewards=n_rewards)

    sensors = np.array([0.0, 0.7, 0.1])
    rewards = [0.0, 0.0]
    model.update_sensors_and_rewards(sensors, rewards)

    actions = np.array([0.0, 1.0, 0.0, 0.0])
    model.update_actions(actions)

    sensors = np.array([0.5, 0.0, 0.3])
    rewards = [None, 0.6]
    model.update_sensors_and_rewards(sensors, rewards)

    assert model.s_a_occurrences[1][1] == 0.7
    assert model.s_a_occurrences[1][0] == 0.0
    assert model.s_a_occurrences[0][1] == 0.0

    assert model.s_a_s_occurrences[1][1][0] == 0.35
    assert model.s_a_s_occurrences[0][1][0] == 0.0
    assert model.s_a_s_occurrences[1][0][0] == 0.0
    assert model.s_a_s_occurrences[1][1][1] == 0.0

    assert model.s_a_rewards[1][1][0] == 0
    assert model.s_a_rewards[1][1][1] > 0
    assert model.s_a_rewards[1][0][1] == 0
    assert model.s_a_rewards[0][1][1] == 0

    assert model.s_a_activities[1][0] == 0.0
    assert model.s_a_activities[0][1] == 0.0
    assert model.s_a_activities[1][1] == 0.7
    assert model.s_a_activities[2][1] == 0.1
    assert model.s_a_activities[0][3] == 0.0
    assert model.s_a_activities[1][3] == 0.0
    assert model.s_a_activities[2][3] == 0.0
    assert model.s_a_activities[1][4] == 0.0

    assert model.s_a_uncertainties[1][1] < 0.7
    assert model.s_a_uncertainties[0][1] == 1.0


def test_predictions():
    n_sensors = 3
    n_actions = 4
    n_rewards = 2
    model = Model(n_sensors=n_sensors, n_actions=n_actions, n_rewards=n_rewards)

    sensors = np.array([0.0, 0.7, 0.1])
    rewards = [0.0, 0.0]
    model.update_sensors_and_rewards(sensors, rewards)

    actions = np.array([0.0, 1.0, 0.0, 0.0])
    model.update_actions(actions)

    sensors = np.array([0.5, 0.0, 0.3])
    rewards = [None, 0.6]
    model.update_sensors_and_rewards(sensors, rewards)

    predictions, rewards, uncertainties = model.predict()

    assert predictions[0][0] == 0.0
    assert predictions[1][0] > 0.0
    assert predictions[2][0] == 0.0

    assert predictions[0][1] == 0.0
    assert predictions[1][1] == 0.0
    assert predictions[2][1] == 0.0

    assert predictions[0][2] == 0.0
    assert predictions[1][2] > 0.0
    assert predictions[2][2] == 0.0

    assert rewards[0] == 0.0
    assert rewards[1] > 0.0
    assert rewards[2] == 0.0
    assert rewards[3] == 0.0

    assert 0.4 < uncertainties[0] < 0.6
    assert 0.4 < uncertainties[1] < 0.6
    assert 0.4 < uncertainties[2] < 0.6

    predictions, reward, uncertainty = model.predict(i_action=1)

    assert predictions[0] > 0.0
    assert predictions[1] == 0.0
    assert predictions[2] > 0.0

    assert reward > 0.0

    assert 0.4 < uncertainties[0] < 0.6
    assert 0.4 < uncertainties[1] < 0.6
    assert 0.4 < uncertainties[2] < 0.6

    sensors = np.array([0.0, 1.0, 0.0])
    predictions, rewards, uncertainties = model.predict(sensors=sensors)

    assert predictions[0][0] == 0.0
    assert predictions[1][0] > 0.0
    assert predictions[2][0] == 0.0

    assert predictions[0][1] == 0.0
    assert predictions[1][1] == 0.0
    assert predictions[2][1] == 0.0

    assert predictions[0][2] == 0.0
    assert predictions[1][2] > 0.0
    assert predictions[2][2] == 0.0

    assert rewards[0] == 0.0
    assert rewards[1] > 0.0
    assert rewards[2] == 0.0
    assert rewards[3] == 0.0

    assert uncertainties[0] == 1.0
    assert 0.4 < uncertainties[1] < 0.6
    assert uncertainties[2] == 1.0

    predictions, reward, uncertainty = model.predict(sensors=sensors, i_action=1)

    assert predictions[0] > 0.0
    assert predictions[1] == 0.0
    assert predictions[2] > 0.0

    assert reward > 0.0

    assert uncertainties[0] == 1.0
    assert 0.4 < uncertainties[1] < 0.6
    assert uncertainties[2] == 1.0


# def test_plan_one_step():
# def test_plan_two_steps():
