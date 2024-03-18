import numpy as np


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
        decay_rate=0.35,
        reward_update_rate=.3,
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

        # The rate at which feature activity and
        # feature-action pair activity decays between time steps
        # for the purpose of estimating and attributing reward and
        # predicting and modeling the outcome.
        self.decay_rate = decay_rate

        # The rate at which new observations of reward are incorporated into
        # the running estimate.
        self.reward_update_rate = reward_update_rate

    def predict(self, feature_activities, rewards):
        # The most recent batch of feature activities is also the outcome
        # of the feature-action (s-a) pairs that are currently active
        outcome_activities = feature_activities

        # Update feature activities.
        # This includes a decayed version of previous activities, combined with
        # the most recent set of observed feature activities.
        aged_activities = self.feature_activities * (1 - self.decay_rate)
        self.feature_activities = np.maximum(feature_activities, aged_activities)

        # Update transition counts and probabilities
        new_s_a_s_occurrences = (
            self.s_a_activities[:, :, np.newaxis]
            * outcome_activities[np.newaxis, np.newaxis, :]
        )
        self.s_a_s_occurrences = self.s_a_s_occurrences + new_s_a_s_occurrences
        s_a_s_likelihoods = (
            self.s_a_s_occurrences / (self.s_a_occurrences[:, :, np.newaxis] + 1))

        # Update the reward
        # The logic for handling single or multiple rewards is buried in this function.
        self.update_reward(rewards)

        # Predict outcomes
        # Conditional predictions represent "what if" scenarios for every
        # possible action. It has the shape (n_actions, n_outcomes)
        conditional_predictions = np.max(
            self.feature_activities[:, np.newaxis, np.newaxis] * s_a_s_likelihoods,
            axis=0,
        )

        # Predict reward
        # Start with a weighted average of all expected rewards, where the
        # weights are the feature activities.
        epsilon = 1e-12
        conditional_multi_rewards = np.sum(
            self.feature_activities[:, np.newaxis, np.newaxis] * self.s_a_rewards,
            axis=0
        ) / (np.sum(self.feature_activities) + epsilon)
        conditional_rewards = np.sum(conditional_multi_rewards, axis=1)

        print("rewards")
        print(self.s_a_rewards[:, :3])
        print("conditional multi rewards")
        print(conditional_multi_rewards)
        print("conditional rewards")
        print(conditional_rewards)

        # Find uncertainty associated with each action
        uncertainties = np.max(
            self.feature_activities[:, np.newaxis] * self.s_a_uncertainties, axis=0
        )

        return conditional_predictions, conditional_rewards, uncertainties

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
        self.s_a_activities = self.s_a_activities * (1 - self.decay_rate)
        new_s_a_activities = (
            self.feature_activities[:, np.newaxis] * self.actions[np.newaxis, :]
        )
        self.s_a_activities = np.maximum(new_s_a_activities, self.s_a_activities)
        self.s_a_occurrences = self.s_a_occurrences + self.s_a_activities
        self.s_a_uncertainties = 1 / (1 + self.s_a_occurrences)

    def update_reward(self, rewards):
        if self.n_rewards > 1:
            rewards = list(rewards)
            for i_reward, reward in enumerate(rewards):
                if reward is not None:
                    delta_reward = reward - self.s_a_rewards[:, :, i_reward]
                    self.s_a_rewards[:, :, i_reward] += (
                        delta_reward *
                        self.s_a_activities *
                        self.reward_update_rate
                    )
        else:
            if rewards is not None:
                i_reward = 0
                delta_reward = rewards - self.s_a_rewards[:, :, i_reward]
                self.s_a_rewards[:, :, i_reward] += (
                    delta_reward *
                    self.s_a_activities *
                    self.reward_update_rate
                )
