import numpy as np
from numba import njit


class NaiveCartographer(object):
    """
    The "model" part of a model-based reinforcement learning algorithm.

    There is more detail in the README, and
    you can dig deeper into how FNC works and why
    in the white paper at https://brandonrohrer.com/cartographer
    """

    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=1,
        feature_decay_rate=1.0,
        trace_decay_rate=0.3,
        reward_update_rate=0.01,  # a.k.a. learning_rate
    ):
        self.n_features = n_sensors
        self.feature_activities = np.zeros(self.n_features)
        self.n_outcomes = self.n_features

        # Add 2 special actions that are internal to the model,
        # a "do-nothing" action and an "average" action.
        # The "average" action is always on and helps FNC learn
        # which outcomes and rewards are independent of the actions
        # it takes. "do-nothing" is exactly what it sounds like.
        self.n_actions = n_actions + 2
        self.actions = np.zeros(self.n_actions)

        # FNC allows for multiple rewards. This accommodates, say,
        # an internal and an external reward. If a reward is
        # None on a given timestep, it's not updated.
        self.n_rewards = n_rewards

        # The properties associated with each
        # feature-action (state-action or s_a) pair and
        # feature-action-outcome (state-action-state or s_a_s) transition.
        # If M is the number of features and N is the number of actions,
        # the size of 2D arrays is M * N and the shape of
        # 3D arrays is M**2 * N. As a heads up, this can eat up
        # memory as M gets large. They are indexed as follows:
        #     index 0 : s_1 (previous feature)
        #     index 1 : a (action)
        #     index 2 : s_2 (outcome feature)
        # The s_a arrays can be 2D because they lack
        # information about the resulting feature outcomes.
        s_a_size = (self.n_features, self.n_actions)
        self.s_a_activities = np.zeros(s_a_size)
        self.s_a_occurrences = np.zeros(s_a_size)
        self.s_a_uncertainties = np.zeros(s_a_size)

        rewards_size = (self.n_features, self.n_actions, self.n_rewards)
        self.s_a_rewards = np.zeros(rewards_size)

        transition_size = (self.n_features, self.n_actions, self.n_outcomes)
        self.s_a_s_occurrences = np.zeros(transition_size)

        # The rate at which feature activity decays between time steps.
        # If it's less than one, it lets feature activity persist somewhat.
        self.feature_decay_rate = feature_decay_rate

        # The rate at which feature-action pair activity decays between time steps
        # for the purpose of estimating and attributing reward and
        # predicting and modeling the outcome.
        self.trace_decay_rate = trace_decay_rate

        # The rate at which new observations of reward are incorporated into
        # the running estimate.
        self.reward_update_rate = reward_update_rate

        self.activity_threshold = 0.1
        self.epsilon = 1e-12

    def update_sensors_and_rewards(self, feature_activities, rewards):
        # The most recent batch of feature activities is also the outcome
        # of the feature-action (s-a) pairs that are currently active
        outcome_activities = feature_activities

        # Update feature activities.
        # This includes a decayed version of previous activities, combined with
        # the most recent set of observed feature activities.
        aged_activities = self.feature_activities * (1 - self.feature_decay_rate)
        self.feature_activities = np.maximum(feature_activities, aged_activities)

        # Update transition counts and probabilities.
        i_sensors, i_actions = np.where(self.s_a_activities > self.activity_threshold)
        i_outcomes = np.where(outcome_activities > self.activity_threshold)[0]
        if i_sensors.size > 0 and i_actions.size > 0 and i_outcomes.size > 0:
            for i_sensor, i_action in zip(i_sensors, i_actions):
                for i_outcome in i_outcomes:
                    self.s_a_s_occurrences[i_sensor, i_action, i_outcome] += (
                        self.s_a_activities[i_sensor, i_action]
                        * outcome_activities[i_outcome]
                    )

        # Update the reward.
        # Handles either single or multiple rewards.

        # Adding uncertainties to the reward update rate has the affect of
        # accelerating reward learning in early iterations when there
        # is a dearth of information. It speeds up initial learning
        # considerably, but still allows the model to fall into
        # more incremental reward learning as it matures.
        reward_update_rate = self.reward_update_rate + self.s_a_uncertainties

        if self.n_rewards > 1:
            # Multiple reward case
            rewards = list(rewards)
            for i_reward, reward in enumerate(rewards):
                if reward is not None:
                    delta_reward = reward - self.s_a_rewards[:, :, i_reward]
                    self.s_a_rewards[:, :, i_reward] += (
                        delta_reward * self.s_a_activities * reward_update_rate
                    )
        else:
            # Single reward case
            if rewards is not None:
                i_reward = 0
                delta_reward = rewards - self.s_a_rewards[:, :, i_reward]
                self.s_a_rewards[:, :, i_reward] += (
                    delta_reward * self.s_a_activities * reward_update_rate
                )

    def update_actions(self, actions):
        # As a sensor, the do-nothing feature can only be 1
        # when there is no other activity from any of the other
        # actions or goals.
        total_action = np.sum(actions)
        inaction = max(1 - total_action, 0)
        # The average action is always set to 1.
        # This allows is to generate the average response.
        average_action = 1
        self.actions = np.concatenate((actions, np.array([inaction, average_action])))

        # Update feature-action pairs
        self.s_a_activities = self.s_a_activities * (1 - self.trace_decay_rate)
        new_s_a_activities = (
            self.feature_activities[:, np.newaxis] * self.actions[np.newaxis, :]
        )
        self.s_a_activities = np.maximum(new_s_a_activities, self.s_a_activities)
        self.s_a_occurrences += self.s_a_activities
        self.s_a_uncertainties = 1 / (1 + self.s_a_occurrences)

    def predict(self, sensors=None, i_action=None):
        """
        Predict the next likely sensor values, rewards, and their uncertainties.

        If sensor values are not provided, they are assumed to be the current
        set of feature activities.

        If an action index is not provided, the predictions for all available
        actions are calculated (a full set of conditional predictions).
        """
        if sensors is None:
            sensors = self.feature_activities
        else:
            assert sensors.size == self.n_features

        if i_action is None:
            return self._get_conditional_predictions(sensors)
        else:
            return self._get_predictions(sensors, i_action)

    def _get_conditional_predictions(self, sensors):
        # Conditional predictions represent "what if" scenarios for every
        # possible action. It has the shape (n_actions, n_outcomes)
        conditional_predictions = np.zeros((self.n_actions, self.n_outcomes))
        make_conditional_predictions(
            self.activity_threshold,
            conditional_predictions,
            sensors,
            self.s_a_s_occurrences,
            self.s_a_occurrences,
        )

        # Predict reward
        #
        # Method 1: Weighted average
        # Start with a weighted average of all expected rewards, where the
        # weights are the feature activities.
        # conditional_multi_rewards = np.sum(
        #     sensors[:, np.newaxis, np.newaxis] * self.s_a_rewards, axis=0
        # ) / (np.sum(sensors) + self.epsilon)
        #
        # Method 2: Maximum
        # Find the maximum expected reward across all features.
        # This works well on the One-hot conditional bandit.
        conditional_multi_rewards = np.max(
            sensors[:, np.newaxis, np.newaxis] * self.s_a_rewards, axis=0
        )
        # Finally, sum across all rewards.
        conditional_rewards = np.sum(conditional_multi_rewards, axis=1)

        # Find uncertainty associated with each action
        uncertainties = np.max(sensors[:, np.newaxis] * self.s_a_uncertainties, axis=0)

        return conditional_predictions, conditional_rewards, uncertainties

    def _get_predictions(self, sensors, i_action):
        # These predictions represent "what if" scenarios for a particular action.
        # It is a one-dimensional array of predicted outcome feature activities.
        s_a_s_likelihoods = self.s_a_s_occurrences[:, i_action, :] / (
            self.s_a_occurrences[:, i_action, np.newaxis] + 1
        )
        predictions = np.max(sensors[:, np.newaxis] * s_a_s_likelihoods, axis=0)

        # Predict reward.
        # Start with a weighted average of all expected rewards, where the
        # weights are the feature activities.
        multi_rewards = np.sum(
            sensors[:, np.newaxis] * self.s_a_rewards[:, i_action, :], axis=0
        ) / (np.sum(sensors) + self.epsilon)
        predicted_reward = np.sum(multi_rewards)

        # Find uncertainty associated with each action
        uncertainty = np.max(sensors * self.s_a_uncertainties[:, i_action])

        return predictions, predicted_reward, uncertainty


@njit
def make_conditional_predictions(
    activity_threshold,
    predictions,
    sensors,
    s_a_s_occurrences,
    s_a_occurrences,
):
    n_features, n_actions, n_outcomes = s_a_s_occurrences.shape
    i_sensors = np.where(sensors > activity_threshold)[0]
    for i_action in range(n_actions):
        for i_outcome in range(n_outcomes):
            max_val = -1e10
            for i_sensor in i_sensors:
                max_val = max(
                    max_val,
                    sensors[i_sensor]
                    * s_a_s_occurrences[i_sensor, i_action, i_outcome]
                    / (s_a_occurrences[i_sensor, i_action] + 1),
                )
            predictions[i_action, i_outcome] = max_val
