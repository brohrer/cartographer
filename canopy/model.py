from numba import njit
import numpy as np
import becca.model_numba as nb

EPSILON = 1e-12


class CanopyModel(object):
    """
    Build a predictive model based on sequences of features, goals, reward.

    Canopy is meant to be the "model" part of a
    model-based reinforcement learning algorithm.
    It builds a Markov Decision Process model of
    its world in the form of a set of sequences.
    It builds a value function too. It creates a set of feature-goal pairs
    and associates a reward
    and curiosity with each. This is analogous to the state-action
    value functions of Q-learning.

    A model and a value function allow for prediction and planning.
    When paired with a planner, these make a complete
    reinforcement learning algorithm.

    Prediction.
    Knowing the current active features
    and recent goals, both the reward and the resulting features can be
    anticipated.

    Planning.
    Feature-goal-feature tuples can
    be chained together to formulate multi-step plans while maximizing
    reward and probability of successfully reaching the goal.

    """
    def __init__(
        self,
        brain=None,
        debug=False,
        n_features=0,
        credit_decay_rate=.35,
        curiosity_update_rate=.01,
        prefix_decay_rate=.5,
        reward_update_rate=.03,
    ):
        """
        Parameters
        ----------
        brain : Brain
            The Brain to which this model belongs. Some of the brain's
            parameters are useful in initializing the model.
        n_features : int
            The total number of features allowed in this model.
        credit_decay_rate : float
            The rate at which the trace, a prefix's credit for the
            future reward, decays with each time step.
        curiosity_update_rate : float
            One of the factors that determines he rate at which
            a prefix increases its curiosity.
        prefix_decay_rate : float
            The rate at which prefix activity decays between time steps
            for the purpose of calculating reward and finding the outcome.
        reward_update_rate : float
            The rate at which a prefix modifies its reward estimate
            based on new observations.
        """
        self.debug = debug

        # n_features : int
        #     The maximum number of features that the model can expect
        #     to incorporate. Knowing this allows the model to
        #     pre-allocate all the data structures it will need.
        #     Add 2 features/goals that are internal to the model,
        #     An "always on" and a "nothing else is on".
        self.n_features = n_features + 2

        # previous_feature_activities,
        # feature_activities : array of floats
        #     Features are characterized by their
        #     activity, that is, their level of activation at each time step.
        #     Activity can vary between zero and one.
        self.previous_feature_activities = np.zeros(self.n_features)
        self.feature_activities = np.zeros(self.n_features)
        # feature_fitness : array of floats
        #     The predictive fitness of each feature is regularly updated.
        #     This helps determine which features to keep and which to
        #     swap out for new candidates.
        self.feature_fitness = np.zeros(self.n_features)

        # goal_activities: array of floats
        #     Goals can be set for features.
        #     They are temporary incentives, used for planning and
        #     goal selection. These can vary between zero and one.
        #     Votes are used to help choose a new goal each time step.
        self.goal_activities = np.zeros(self.n_features)

        # prefix_curiosities,
        # prefix_occurrences,
        # prefix_activities,
        # prefix_rewards : 2D array of floats
        # sequence_occurrences : 3D array of floats
        #     The properties associated with each sequence and prefix.
        #     If N is the number of features,
        #     the size of 2D arrays is N**2 and the shape of
        #     3D arrays is N**3. As a heads up, this can eat up
        #     memory as M gets large. They are indexed as follows:
        #         index 0 : feature_1 (past or pre-feature)
        #         index 1 : feature_goal
        #         index 2 : feature_2 (future or post-feature)
        #     The prefix arrays can be 2D because they lack
        #     information about the resulting feature.
        _2D_size = (self.n_features, self.n_features)
        _3D_size = (self.n_features, self.n_features, self.n_features)
        # Making believe that everything has occurred once in the past
        # makes it easy to believe that it might happen again in the future.
        self.conditional_rewards = np.zeros(self.n_features)
        self.conditional_curiosities = np.zeros(self.n_features)
        self.conditional_predictions = np.zeros(_2D_size)
        self.prefix_activities = np.zeros(_2D_size)
        self.prefix_credit = np.zeros(_2D_size)
        self.prefix_occurrences = np.zeros(_2D_size)
        self.prefix_curiosities = np.zeros(_2D_size)
        self.prefix_rewards = np.zeros(_2D_size)
        self.prefix_uncertainties = np.zeros(_2D_size)
        self.sequence_occurrences = np.zeros(_3D_size)
        self.sequence_likelihoods = np.zeros(_3D_size)

        self.credit_decay_rate = credit_decay_rate
        self.curiosity_update_rate = curiosity_update_rate
        self.prefix_decay_rate = prefix_decay_rate
        self.reward_update_rated = reward_update_rate

    def step(self, feature_activities, reward):
        """
        Update the model and choose a new goal.

        Parameters
        ----------
        feature_activities : array of floats
            The current activity levels of each of the feature candidates.
        reward : float
            The reward reported by the world during
            the most recent time step.
        """
        # Update feature_activities and previous_feature_activities
        self.update_activities(feature_activities)

        # Update sequences before prefixes.
        nb_update_prefixes(
            self.prefix_decay_rate,
            self.previous_feature_activities,
            self.goal_activities,
            self.prefix_activities,
            self.prefix_occurrences,
            self.prefix_uncertainties,
        )

        nb_update_sequences(
            self.feature_activities,
            self.prefix_activities,
            self.prefix_occurrences,
            self.sequence_occurrences,
            self.sequence_likelihoods,
        )

        nb_update_rewards(
            self.reward_update_rate,
            reward,
            self.prefix_credit,
            self.prefix_rewards,
        )

        nb_update_curiosities(
            self.curiosity_update_rate,
            self.prefix_occurrences,
            self.prefix_curiosities,
            self.previous_feature_activities,
            self.feature_activities,
            self.goal_activities,
            self.prefix_uncertainties,
        )

        nb_predict_features(
            self.feature_activities,
            self.sequence_likelihoods,
            self.conditional_predictions
        )

        nb_predict_rewards(
            self.feature_activities,
            self.prefix_rewards,
            self.conditional_rewards,
        )

        nb_predict_curiosities(
            self.feature_activities,
            self.prefix_curiosities,
            self.conditional_curiosities,
        )

        return (
            self.feature_activities,
            self.conditional_predictions,
            self.conditional_rewards,
            self.conditional_curiosities)

    def update_activities(self, feature_activities):
        """
        Apply new activities,

        Parameters
        ----------
        feature_activities: array of floats

        Returns
        -------
        None, but updates class members
        feature_activities: array of floats
        previous_feature_activities: array of floats
        """
        # Augment the feature_activities with the two internal features,
        # the "always on" (index of 0) and
        # the "null" or "nothing else is on" (index of 1).
        self.previous_feature_activities = self.feature_activities
        self.feature_activities = np.concatenate((
            np.zeros(2), feature_activities))
        self.feature_activities[0] = 1
        total_activity = np.sum(self.feature_activities[2:])
        inactivity = max(1 - total_activity, 0)
        self.feature_activities[1] = inactivity

    def update_goals(self, goals, i_new_goal):
        """
        Given a set of goals, record and execute them.

        Parameters
        ----------
        goals: array of floats
            The current set of goal activities.
        i_new_goal: int
            The index of the most recent feature goal.

        Returns
        -------
        feature_pool_goals: array of floats
        """
        self.goal_activities = goals
        nb_update_reward_credit(
            i_new_goal,
            self.feature_activities,
            self.credit_decay_rate,
            self.prefix_credit)

    def calculate_fitness(self):
        """
        Calculate the predictive fitness of all the feature candidates.

        Returns
        -------
        feature_fitness: array of floats
            The fitness of each of the feature candidate inputs to
            the model.
        """
        nb_update_fitness(
            self.feature_fitness,
            self.prefix_occurrences,
            self.prefix_rewards,
            self.prefix_uncertainties,
            self.sequence_occurrences)

        return self.feature_fitness

    def reset_inputs(self, resets):
        """
        Add and reset feature inputs as appropriate.

        Parameters
        ----------
        upstream_resets: array of ints
            Indices of the feature candidates to reset.

        Returns
        -------
        resets: array of ints
            Indices of the features that were reset.
        """
        # Account for the model's 2 internal features.
        model_resets = [i_reset + 2 for i_reset in resets]
        # Reset features throughout the model.
        # It's like they never existed.
        for i in model_resets:
            self.previous_feature_activities[i] = 0
            self.feature_activities[i] = 0
            self.feature_fitness[i] = 0
            self.goal_activities[i] = 0
            self.prefix_activities[i, :] = 0
            self.prefix_activities[:, i] = 0
            self.prefix_credit[i, :] = 0
            self.prefix_credit[:, i] = 0
            self.prefix_occurrences[i, :] = 0
            self.prefix_occurrences[:, i] = 0
            self.prefix_curiosities[i, :] = 0
            self.prefix_curiosities[:, i] = 0
            self.prefix_rewards[i, :] = 0
            self.prefix_rewards[:, i] = 0
            self.prefix_uncertainties[i, :] = 0
            self.prefix_uncertainties[:, i] = 0
            self.sequence_occurrences[i, :, :] = 0
            self.sequence_occurrences[:, i, :] = 0
            self.sequence_occurrences[:, :, i] = 0
            self.sequence_likelihoods[i, :, :] = 0
            self.sequence_likelihoods[:, i, :] = 0
            self.sequence_likelihoods[:, :, i] = 0


@njit
def nb_update_prefixes(
    prefix_decay_rate,
    previous_feature_activities,
    goal_activities,
    prefix_activities,
    prefix_occurrences,
    prefix_uncertainties,
):
    """
    Update the activities and occurrences of the prefixes.

    The new activity of a feature-goal prefix, n,  is
         n = f * g, where
    f is the previous_feature_activities and
    g is the current goal_increase.

    p, the prefix activity, is a decayed version of n.
    """
    n_features, n_goals = prefix_activities.shape
    for i_feature in range(n_features):
        for i_goal in range(n_goals):
            prefix_activities[i_feature, i_goal] *= 1 - prefix_decay_rate

            new_prefix_activity = (
                previous_feature_activities[i_feature] *
                goal_activities[i_goal]
            )
            prefix_activities[i_feature, i_goal] += new_prefix_activity
            prefix_activities[i_feature, i_goal] = min(
                prefix_activities[i_feature, i_goal], 1)

            # Increment the lifetime sum of prefix activity.
            prefix_occurrences[i_feature, i_goal] += (
                prefix_activities[i_feature, i_goal])

            # Adjust uncertainty accordingly.
            prefix_uncertainties[i_feature, i_goal] = 1 / (
                1 + prefix_occurrences[i_feature, i_goal])


@njit
def nb_update_sequences(
    feature_activities,
    prefix_activities,
    prefix_occurrences,
    sequence_occurrences,
    sequence_likelihoods,
):
    """
    Update the number of occurrences of each sequence.

    The new sequence activity, n, is
        n = p * f, where
    p is the prefix activities from the previous time step and
    f is the outcome_activities

    For any given sequence, the probability (sequence_likelihood)
    it will occur on the next time
    step, given the associated goal is selected, is

        f * s / (p + 1)

    where

        f: feature activity
        s: number of sequence occurrences
        p: number of prefix occurrences

    Adding 1 to p prevents badly behaved fractions.
    """
    n_features, n_goals, n_outcomes = sequence_occurrences.shape
    # Iterate over outcomes
    for i_outcome in range(n_outcomes):
        if feature_activities[i_outcome] > EPSILON:
            # Iterate over goals and pre-features
            for i_goal in range(n_goals):
                for i_feature in range(n_features):
                    # These are still the prefix activities from the
                    # previous time step. Together with the current
                    # feature activities, these constitute sequences.
                    if prefix_activities[i_feature, i_goal] > EPSILON:
                        sequence_occurrences[
                            i_feature, i_goal, i_outcome] += (
                            prefix_activities[i_feature, i_goal] *
                            feature_activities[i_outcome])
                        occurrences = sequence_occurrences[
                                i_feature, i_goal, i_outcome]
                        opportunities = prefix_occurrences[
                                i_feature, i_goal] + 1
                        sequence_likelihoods[
                            i_feature, i_goal, i_outcome] = (
                            occurrences / opportunities)


@njit
def nb_update_rewards(
    reward_update_rate,
    reward,
    prefix_credit,
    prefix_rewards,
):
    """
    Assign credit for the current reward to any recently active prefixes.

    Increment the expected reward associated with each prefix.
    The size of the increment is larger when:
        1. the discrepancy between the previously learned and
            observed reward values is larger and
        2. the prefix activity is greater.
    Another way to say this is:
    If either the reward discrepancy is very small
    or the sequence activity is very small, there is no change.
    """
    n_features, n_goals = prefix_rewards.shape
    for i_feature in range(n_features):
        for i_goal in range(n_goals):
            # credit: How much responsibility for this reward is assigned to
            # this prefix?
            credit = prefix_credit[i_feature, i_goal]
            if credit > EPSILON:
                # delta: How big is the discrepancy between
                # the observed reward and what has been seen preivously.
                delta = reward - prefix_rewards[i_feature, i_goal]
                if reward > prefix_rewards[i_feature, i_goal]:
                    update_scale = .5
                else:
                    update_scale = 1
                prefix_rewards[i_feature, i_goal] += (
                    delta * credit * reward_update_rate * update_scale)


@njit
def nb_update_curiosities(
    curiosity_update_rate,
    prefix_occurrences,
    prefix_curiosities,
    previous_feature_activities,
    feature_activities,
    goal_activities,
    prefix_uncertainties,
):
    """
    Use a collection of factors to increment the curiosity for each prefix.
    """
    n_features, n_goals = prefix_curiosities.shape
    for i_feature in range(n_features):
        for i_goal in range(n_goals):

            # Fulfill curiosity on the previous time step's goals.
            curiosity_fulfillment = (previous_feature_activities[i_feature] *
                                     goal_activities[i_goal])
            prefix_curiosities[i_feature, i_goal] -= curiosity_fulfillment
            prefix_curiosities[i_feature, i_goal] = max(
                prefix_curiosities[i_feature, i_goal], 0)

            # Increment the curiosity based on several multiplicative
            # factors.
            #     curiosity_update_rate : a constant
            #     uncertainty : an estimate of how much is not yet
            #         known about this prefix. It is a function of
            #         the total past occurrences.
            #     feature_activities : The activity of the prefix's feature.
            #         Only increase the curiosity if the feature
            #         corresponding to the prefix is active.
            prefix_curiosities[i_feature, i_goal] += (
                curiosity_update_rate *
                prefix_uncertainties[i_feature, i_goal] *
                feature_activities[i_feature])


@njit
def nb_predict_features(
    feature_activities,
    sequence_likelihoods,
    conditional_predictions
):
    """
    Make a prediction about which features are going to become active soon,
    conditional on which goals are chosen.

    Parameters
    ----------
    feature_activities: array of floats
    sequence_likelihoods: 3D array of floats
    conditional_predictions: 2D array of floats
        This is updated to represent the new predictions for this time step.
    """
    n_features, n_goals, n_outcomes = sequence_likelihoods.shape
    conditional_predictions = np.zeros((n_goals, n_outcomes))
    for i_feature in range(n_features):
        if feature_activities[i_feature] > EPSILON:
            do_nothing_prediction = sequence_likelihoods[i_feature, 1, :]
            # for i_outcome in range(2, n_outcomes):
            #     if (do_nothing_prediction[i_outcome] >
            #             conditional_predictions[1, i_outcome]):
            #         conditional_predictions[1, i_outcome] = (
            #             do_nothing_prediction[i_outcome])
            for i_goal in range(1, n_goals):
                for i_outcome in range(1, n_outcomes):
                    p_sequence = (sequence_likelihoods[
                        i_feature, i_goal, i_outcome] -
                        do_nothing_prediction[i_outcome])
                    if (p_sequence >
                            conditional_predictions[i_goal, i_outcome]):
                        conditional_predictions[i_goal, i_outcome] = (
                            p_sequence)


@njit
def nb_predict_rewards(
    feature_activities,
    prefix_rewards,
    conditional_rewards,
):
    """
    Make a prediction about how much reward will result from each goal.

    For any given prefix, the reward expected to occur on the next time
    step, given the associated goal is selected, is

        f * r

    where

        f: feature activity
        r: prefix rewards

    Parameters
    ----------
    feature_activities: array of floats
    prefix_rewards: 2D array of floats
    conditional_rewards: 2D array of floats
        This is updated to represent the new reward predictions
        for this time step.
    """
    n_features, n_goals = prefix_rewards.shape
    conditional_rewards = np.zeros(n_goals)
    for i_feature in range(2, n_features):
        # Goal[1] is the special "do nothing" goal. It helps to distinguish
        # between reward that is due to a goal and reward that would have
        # been received even if doing nothing.
        do_nothing_reward = prefix_rewards[i_feature, 1]

        # Calculate the expected change in reward for each goal,
        # compared to doing nothing.
        for i_goal in range(1, n_goals):
            expected_reward = (
                feature_activities[i_feature]
                * prefix_rewards[i_feature, i_goal] - do_nothing_reward)
            if expected_reward > conditional_rewards[i_goal]:
                conditional_rewards[i_goal] = expected_reward


@njit
def nb_predict_curiosities(
    feature_activities,
    prefix_curiosities,
    conditional_curiosities,
):
    """
    Make a prediction about how much reward will result from each goal.

    For any given prefix, the reward expected to occur on the next time
    step, given the associated goal is selected, is

        f * r

    where

        f: feature activity
        r: prefix curiosities

    Parameters
    ----------
    feature_activities: array of floats
    prefix_curiosities: 2D array of floats
    conditional_curiosities: 2D array of floats
        This is updated to represent the new reward predictions
        for this time step.
    """
    n_features, n_goals = prefix_curiosities.shape
    conditional_curiosities = np.zeros(n_goals)
    for i_feature in range(n_features):
        # Ignore the first "always on" feature.
        # It doesn't do anything as a goal.
        for i_goal in range(1, n_goals):
            expected_curiosity = (
                feature_activities[i_feature]
                * prefix_curiosities[i_feature, i_goal])
            if expected_curiosity > conditional_curiosities[i_goal]:
                conditional_curiosities[i_goal] = expected_curiosity


# This makes use of vectorized numpy calls, so numba is unneccessary here.
def nb_update_fitness(
    feature_fitness,
    prefix_occurrences,
    prefix_rewards,
    prefix_uncertainties,
    sequence_occurrences,
):
    """
    Calculate the fitness of each feature

    Parameters
    ----------
    feature_fitness: array of floats
        The fitness score as of this time step for each of the feature
        inputs to the model. This is modified with calculated values.
    prefix_occurrences: 2D array of floats
    prefix_rewards: 2D array of floats
    prefix_uncertainties: 2D array of floats
    sequence_occurrences: 3D array of floats
    """
    # Calculate the ability of each prefix to predict the features that
    # follow it.
    # Base it on the single most successfully predicted sequence.
    # TODO: debug numba type inference error in this line
    outcome_prediction_score = (
        np.max(sequence_occurrences, axis=2) /
        (prefix_occurrences + EPSILON))
    # Calculate the ability of each prefix to predict reward or punishment.
    reward_prediction_score = np.abs(prefix_rewards)
    prefix_score = outcome_prediction_score + reward_prediction_score
    # Scale fitness by confidence (1 - uncertainty)
    prefix_fitness = prefix_score * (1 - prefix_uncertainties)
    # Find the maximum fitness for each feature across all prefixes,
    # whether as a feature or as a goal.
    feature_fitness = np.max(prefix_fitness, axis=1)
    goal_fitness = np.max(prefix_fitness, axis=0)
    feature_fitness = np.maximum(  # noqa: F841
        feature_fitness, goal_fitness)


@njit
def nb_update_reward_credit(
    i_new_goal,
    feature_activities,
    credit_decay_rate,
    prefix_credit,
):
    """
    Update the credit due each prefix for upcoming reward.
    """
    # Age the prefix credit.
    n_features, n_goals = prefix_credit.shape
    for i_feature in range(n_features):
        for i_goal in range(n_goals):
            # Exponential discounting
            prefix_credit[i_feature, i_goal] *= 1 - credit_decay_rate

    # Update the prefix credit.
    if i_new_goal > -1:
        for i_feature in range(n_features):
            # Accumulation strategy:
            # add new credit to existing credit, with a max of 1.
            prefix_credit[i_feature, i_new_goal] += (
                feature_activities[i_feature])
            prefix_credit[i_feature, i_new_goal] = min(
                prefix_credit[i_feature, i_new_goal], 1)
