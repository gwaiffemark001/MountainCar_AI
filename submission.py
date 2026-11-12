import math, random
from collections import defaultdict
from typing import List, Callable, Tuple, Dict, Any, Optional, Iterable, Set
import gymnasium as gym
import numpy as np

import util
from util import ContinuousGymMdp, state_t, action_t
from custom_mountain_car import CustomMountainCarEnv

############################################################
# Problem 3a
# Implementing Value Iteration on Number Line (from Problem 1)
def value_iteration(succ_and_reward_prob: Dict[Tuple[state_t, action_t], List[Tuple[state_t, float, float]]], discount: float, epsilon: float = 0.001):
    '''
    Given transition probabilities and rewards, computes and returns v and
    the optimal policy pi for each state.
    - succ_and_reward_prob: Dictionary mapping tuples of (state, action) to a list of (next_state, prob, reward) Tuples.
    - Returns: Dictionary mapping each state to an action.
    '''
    # Define a mapping from states to Set[Actions] so we can determine all the actions that can be taken from s.
    # You may find this useful in your approach.
    state_actions = defaultdict(set)
    for state, action in succ_and_reward_prob.keys():
        state_actions[state].add(action)

    def compute_q(v: Dict[state_t, float], state: state_t, action: action_t) -> float:
        # Return q(state, action) based on v(state)
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return sum(
            probability * (reward + discount * v[next_state])
            for next_state, probability, reward in succ_and_reward_prob[(state, action)]
        )
        # END_YOUR_CODE

    def compute_policy(v: Dict[state_t, float]) -> Dict[state_t, action_t]:
        # Return the policy given v.
        # Remember the policy for a state is the action that gives the greatest q-value.
        # IMPORTANT: if multiple actions give the same q-value, choose the largest action number for the policy. 
        # HINT: We only compute policies for states in state_actions.
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return {
            state: max(
                state_actions[state],
                key=lambda action: (compute_q(v, state, action), action),
            )
            for state in state_actions
        }
        # END_YOUR_CODE

    print('Running value_iteration...')
    v = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
    num_iters = 0
    while True:
        new_v = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
        # update v values using the compute_q function above.
        # repeat until the v values for all states do not change by more than epsilon.
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        for state in state_actions:
            new_v[state] = max(
                (compute_q(v, state, action) for action in state_actions[state]),
                default=0.0,
            )
        if all(abs(new_v[state] - v[state]) <= epsilon for state in state_actions):
            v = new_v
            break
        # END_YOUR_CODE
        v = new_v
        num_iters += 1
    v_opt = v
    print(("value_iteration: %d iterations" % num_iters))
    return compute_policy(v_opt)

############################################################
# Problem 3b
# Model-Based Monte Carlo

# Runs value iteration algorithm on the number line Mdp
# and prints out optimal policy for each state.
def run_vi_over_number_line(mdp: util.NumberLineMdp):
    succ_and_reward_prob = {
        (-mdp.n + 1, 1): [(-mdp.n + 2, 0.2, mdp.penalty), (-mdp.n, 0.8, mdp.left_reward)],
        (-mdp.n + 1, 2): [(-mdp.n + 2, 0.3, mdp.penalty), (-mdp.n, 0.7, mdp.left_reward)],
        (mdp.n - 1, 1): [(mdp.n - 2, 0.8, mdp.penalty), (mdp.n, 0.2, mdp.right_reward)],
        (mdp.n - 1, 2): [(mdp.n - 2, 0.7, mdp.penalty), (mdp.n, 0.3, mdp.right_reward)]
    }

    for s in range(-mdp.n + 2, mdp.n - 1):
        succ_and_reward_prob[(s, 1)] = [(s+1, 0.2, mdp.penalty), (s - 1, 0.8, mdp.penalty)]
        succ_and_reward_prob[(s, 2)] = [(s+1, 0.3, mdp.penalty), (s - 1, 0.7, mdp.penalty)]

    pi = value_iteration(succ_and_reward_prob, mdp.discount)
    return pi


class ModelBasedMonteCarlo(util.RlAlgorithm):
    def __init__(self, actions: List[action_t], discount: float, calc_val_iter_every: int = 10000,
                 exploration_prob: float = 0.2,) -> None:
        self.actions = actions
        self.discount = discount
        self.calc_val_iter_every = calc_val_iter_every
        self.exploration_prob = exploration_prob
        self.num_iters = 0

        # (state, action) -> {next_state -> ct} for all next_state
        self.t_counts = defaultdict(lambda: defaultdict(int))
        # (state, action) -> {next_state -> total_reward} for all next_state
        self.r_total = defaultdict(lambda: defaultdict(float))

        self.pi = {} # Optimal policy for each state. state -> action

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |exploration_prob|, take a random action.
    # Should return random action if the given state is not in self.pi.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always follow the policy if available.
    # HINT: Use random.random() (not np.random()) to sample from the uniform distribution [0, 1]
    def get_action(self, state: state_t, explore: bool = True) -> action_t:
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4: # Always explore
            exploration_prob = 1.0
        elif self.num_iters > 1e6: # Lower the exploration probability by a logarithmic factor.
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        if state not in self.pi or (
            explore and random.random() < exploration_prob
        ):
            return random.choice(self.actions)
        return self.pi[state]
        # END_YOUR_CODE

    # We will call this function with (s, a, r, s'), which is used to update t_counts and r_total.
    # For every self.calc_val_iter_every steps, runs value iteration after estimating succ_and_reward_prob.
    def incorporate_feedback(self, state: state_t, action: action_t, reward: int, next_state: state_t, terminal: bool):

        self.t_counts[(state, action)][next_state] += 1
        self.r_total[(state, action)][next_state] += reward

        if self.num_iters % self.calc_val_iter_every == 0:
            # Estimate succ_and_reward_prob based on self.t_counts and self.r_total.
            # Hint 1: prob(s, a, s') = (counts of transition (s,a) -> s') / (total transtions from (s,a))
            # Hint 2: Reward(s, a, s') = (total reward of (s,a) -> s') / (counts of transition (s,a) -> s')
            # Then run value_iteration and update self.pi.
            succ_and_reward_prob = defaultdict(list)
            # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
            for (state, action), next_state_counts in self.t_counts.items():
                total = sum(next_state_counts.values())
                succ_and_reward_prob[(state, action)] = [
                    (
                        next_state,
                        count / total,
                        self.r_total[(state, action)][next_state] / count,
                    )
                    for next_state, count in next_state_counts.items()
                ]
            self.pi = value_iteration(
                succ_and_reward_prob, self.discount
            )
            # END_YOUR_CODE

############################################################
# Problem 4a
# Performs Tabular q-learning. Read util.RlAlgorithm for more information.
class TabularQLearning(util.RlAlgorithm):
    def __init__(self, actions: List[action_t], discount: float,
                 exploration_prob: float = 0.2, initial_q: float = 0):
        '''
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - exploration_prob: the epsilon value indicating how frequently the policy returns a random action
        - intialQ: the value for intializing q values.
        '''
        self.actions = actions
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.q = defaultdict(lambda: initial_q)
        self.num_iters = 0

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |exploration_prob|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum q-value action.
    # HINT 1: You can access q-value with self.q[state, action]
    # HINT 2: Use random.random() to sample from the uniform distribution [0, 1]
    def get_action(self, state: state_t, explore: bool = True) -> action_t:
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4: # explore
            exploration_prob = 1.0
        elif self.num_iters > 1e5: # Lower the exploration probability by a logarithmic factor.
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        if explore and random.random() < exploration_prob:
            return random.choice(self.actions)
        return max(self.actions, key=lambda action: (self.q[state, action], action))
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def get_step_size(self) -> float:
        return 0.1

    # We will call this function with (s, a, r, s'), which you should use to update |q|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update the q values using self.get_step_size() 
    # HINT 1: The target v for the current state is a combination of the immediate reward
    # and the discounted future value.
    # HINT 2: v for terminal states is 0
    def incorporate_feedback(self, state: state_t, action: action_t, reward: float, next_state: state_t, terminal: bool) -> None:
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        current_q = self.q[state, action]
        future_q = 0.0
        if not terminal:
            future_q = max(self.q[next_state, next_action] for next_action in self.actions)
        target = reward + self.discount * future_q
        self.q[state, action] = current_q + self.get_step_size() * (target - current_q)
        # END_YOUR_CODE

############################################################
# Problem 4b: Fourier feature extractor

def fourier_feature_extractor(
        state: state_t,
        max_coeff: int = 5,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), max_coeff 2, and scale [2, 1, 1], this should output (in any order):
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]
    '''
    if scale is None:
        scale = np.ones_like(state)
    features = None

    # Below, implement the fourier feature extractor as similar to the doc string provided.
    # The return shape should be 1 dimensional ((max_coeff+1)^(len(state)),).
    #
    # HINT: refer to util.polynomial_feature_extractor as a guide for
    # doing efficient arithmetic broadcasting in numpy.

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    scaled_state = np.asarray(state, dtype=float) * np.asarray(scale, dtype=float)
    coefficients = np.indices((max_coeff + 1,) * len(scaled_state)).reshape(
        len(scaled_state), -1
    ).T
    features = np.cos(np.pi * (coefficients @ scaled_state))
    # END_YOUR_CODE

    return features

############################################################
# Problem 4c: q-learning with Function Approximation
# Performs Function Approximation q-learning. Read util.RlAlgorithm for more information.
class FunctionApproxQLearning(util.RlAlgorithm):
    def __init__(self, feature_dim: int, feature_extractor: Callable, actions: List[int],
                 discount: float, exploration_prob=0.2):
        '''
        - feature_dim: the dimensionality of the output of the feature extractor
        - feature_extractor: a function that takes a state and returns a numpy array representing the feature.
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - exploration_prob: the epsilon value indicating how frequently the policy returns a random action
        '''
        self.feature_dim = feature_dim
        self.feature_extractor = feature_extractor
        self.actions = actions
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.w = np.random.standard_normal(size=(feature_dim, len(actions)))
        self.num_iters = 0

    def get_q(self, state: np.ndarray, action: int) -> float:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        return float(np.dot(self.feature_extractor(state), self.w[:, action]))
        # END_YOUR_CODE

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |exploration_prob|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum q-value action.
    # HINT: This function should be the same as your implementation for 4a.
    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4: # Always explore
            exploration_prob = 1.0
        elif self.num_iters > 1e5: # Lower the exploration probability by a logarithmic factor.
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)

        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        if explore and random.random() < exploration_prob:
            return random.choice(self.actions)
        return max(
            self.actions,
            key=lambda action: (self.get_q(state, action), action),
        )
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def get_step_size(self) -> float:
        return 0.005 * (0.99)**(self.num_iters / 500)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update w using self.get_step_size()
    # HINT 1: this part will look similar to 4a, but you are updating self.w
    # HINT 2: review the function approximation module for the update rule
    def incorporate_feedback(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool) -> None:
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        features = self.feature_extractor(state)
        prediction = self.get_q(state, action)
        future_q = 0.0
        if not terminal:
            future_q = max(self.get_q(next_state, next_action) for next_action in self.actions)
        target = reward + self.discount * future_q
        self.w[:, action] += self.get_step_size() * (target - prediction) * features
        # END_YOUR_CODE

############################################################
# Problem 5c: Constrained q-learning

class ConstrainedQLearning(FunctionApproxQLearning):
    def __init__(self, feature_dim: int, feature_extractor: Callable, actions: List[int],
                 discount: float, force: float, gravity: float,
                 max_speed: Optional[float] = None,
                 exploration_prob=0.2):
        super().__init__(feature_dim, feature_extractor, actions,
                         discount, exploration_prob)
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |exploration_prob|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum q-value action that is valid.
    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4: # Always explore
            exploration_prob = 1.0
        elif self.num_iters > 1e5: # Lower the exploration probability by a logarithmic factor.
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)

        # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
        position, velocity = state
        if self.max_speed is None:
            valid_actions = self.actions
        else:
            valid_actions = [
                action
                for action in self.actions
                if velocity
                + (action - 1) * self.force
                - math.cos(3 * position) * self.gravity
                < self.max_speed
            ]
        if not valid_actions:
            return None
        if explore and random.random() < exploration_prob:
            return random.choice(valid_actions)
        return max(
            valid_actions,
            key=lambda action: (self.get_q(state, action), action),
        )
        # END_YOUR_CODE

############################################################
# This is helper code for comparing the predicted optimal
# actions for 2 MDPs with varying max speed constraints
gym.register(
    id="CustomMountainCar-v0",
    entry_point="custom_mountain_car:CustomMountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)

mdp1 = ContinuousGymMdp("CustomMountainCar-v0", discount=0.999, time_limit=1000)
mdp2 = ContinuousGymMdp("CustomMountainCar-v0", discount=0.999, time_limit=1000)

# This is a helper function for 5c. This function runs
# ConstrainedQLearning, then simulates various trajectories through the Mdp
# and compares the frequency of various optimal actions.
def compare_mdp_strategies(mdp1: ContinuousGymMdp, mdp2: ContinuousGymMdp):
    env = mdp1.env.unwrapped
    rl1 = ConstrainedQLearning(
        36,
        lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
        mdp1.actions,
        mdp1.discount,
        env.force,
        env.gravity,
        10000,
        exploration_prob=0.2,
    )
    rl2 = ConstrainedQLearning(
        36,
        lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
        mdp2.actions,
        mdp2.discount,
        env.force,
        env.gravity,
        0.065,
        exploration_prob=0.2,
    )
    sample_krl_trajectories(mdp1, rl1)
    sample_krl_trajectories(mdp2, rl2)

def sample_krl_trajectories(mdp: ContinuousGymMdp, rl: ConstrainedQLearning):
    accelerate_left, no_accelerate, accelerate_right = 0, 0, 0
    for n in range(100):
        traj = util.sample_rl_trajectory(mdp, rl)
        accelerate_left = traj.count(0)
        no_accelerate = traj.count(1)
        accelerate_right = traj.count(2)

    print(f"\nRL with Mdp -> start state:{mdp.start_state()}, max_speed:{rl.max_speed}")
    print(f"  *  total accelerate left actions: {accelerate_left}, total no acceleration actions: {no_accelerate}, total accelerate right actions: {accelerate_right}")
