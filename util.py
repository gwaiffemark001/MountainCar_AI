import collections, random, time
from typing import List, Tuple, Dict, Any, Union, Optional, Iterable
import gymnasium as gym
import numpy as np

state_t = Union[int, float, Tuple[Union[float, int]]]
action_t = Any

def create_bins(low: List[float], high: List[float], num_bins: Union[int, List[int]]) -> List[np.ndarray]:
    """
    Takes in a gym.spaces.Box and returns a set of bins per feature according to num_bins
    """
    assert len(low) == len(high)
    if isinstance(num_bins, int):
        num_bins = [num_bins for _ in range(len(low))]
    assert len(num_bins) == len(low)
    bins = []
    for low, high, n in zip(low, high, num_bins):
        bins.append(np.linspace(low, high, n))
    return bins

def discretize(x, bins) -> Tuple[int]:
    """
    Discretize an array x according to bins
    x: np.ndarray, shape (features,)
    bins: np.ndarray, shape (features, bins)
    """
    return tuple(int(np.digitize(feature, bin)) for feature, bin in zip(x, bins))

# An abstract class representing a Markov Decision Process (Mdp).
class Mdp:
    # Return the start state.
    def start_state(self): raise NotImplementedError("Override me")

    # Property holding the set of possible actions at each state.
    @property
    def actions(self) -> List[action_t]: raise NotImplementedError("Override me")

    # Property holding the discount factor
    @property
    def discount(self): raise NotImplementedError("Override me")

    # property holding the maximum number of steps for running the simulation.
    @property
    def time_limit(self) -> int: raise NotImplementedError("Override me")

    # Transitions the Mdp
    def transition(self, action): raise NotImplementedError("Override me")

class NumberLineMdp(Mdp):
    def __init__(self, left_reward: float = 10, right_reward: float = 50, penalty: float = -5, n: int = 2):
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.penalty = penalty
        self.n = n
        self.terminal_states = {-n, n}

    def start_state(self):
        self.state = 0
        return self.state

    @property
    def actions(self):
        return [1, 2]

    def transition(self, action) -> Tuple[state_t, float, bool]:
        assert self.state not in self.terminal_states, "Attempting to call transition on a terminated Mdp."
        if action == 1:
            forward_prob = 0.2
        elif action == 2:
            forward_prob = 0.3
        else:
            raise ValueError("Invalid Action Provided.")

        if random.random() < forward_prob:
            # Move the agent forward
            self.state += 1
        else:
            # Move the agent backward
            self.state -= 1

        if self.state == self.n:
            reward = self.right_reward
        elif self.state == -self.n:
            reward = self.left_reward
        else:
            reward = self.penalty

        # Check for termination
        terminal = self.state in self.terminal_states

        return (self.state, reward, terminal)

    @property
    def discount(self):
        return 1.0

class GymMdp(Mdp):
    def __init__(self, env, max_speed: Optional[float] = None, discount: float = 0.99, time_limit: Optional[int] = None):
        self.max_speed = max_speed
        if self.max_speed is not None:
            self.env = gym.make(env, max_speed=self.max_speed)
        else:
            self.env = gym.make(env)
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "Must use environments with discrete actions"
        assert isinstance(self.env, gym.wrappers.TimeLimit)
        if time_limit is not None:
            self.env._max_episode_steps = time_limit
        self._time_limit = self.env._max_episode_steps
        self._discount = discount
        self._actions = list(range(self.env.action_space.n))
        self._reset_seed_gen = np.random.default_rng(0)

        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high

    # Return the number of steps before the Mdp should be reset.
    @property
    def time_limit(self) -> int:
        return self._time_limit

    # Return set of actions possible at every state.
    @property
    def actions(self) -> List[action_t]:
        return self._actions

    # Return the Mdp's discount factor
    @property
    def discount(self):
        return self._discount

    # Returns the start state.
    def start_state(self): raise NotImplementedError("Override me")

    # Returns a tuple of (next_state, reward, terminated)
    def transition(self, action): raise NotImplementedError("Override me")

    # Returns custom reward function
    def reward(self, next_state, original_reward):
        if "MountainCar-v0" in self.env.unwrapped.spec.id:
            # reward fn based on x position and velocity
            position_reward = -(self.high[0] - next_state[0])
            velocity_reward = -(self.high[1] - np.abs(next_state[1]))
            return position_reward + velocity_reward
        else:
            return original_reward

class ContinuousGymMdp(GymMdp):

    def start_state(self):
        state, _ = self.env.reset(seed=int(self._reset_seed_gen.integers(0, 1e6)))
        return state

    def transition(self, action):
        next_state, reward, terminal, _, _ = self.env.step(action)
        reward = self.reward(next_state, reward)
        return (next_state, reward, terminal)

class DiscreteGymMdp(GymMdp):

    def __init__(self, env, feature_bins: Union[int, List[int]] = 10, low: Optional[List[float]] = None, high: Optional[List[float]] = None, **kwargs):
        super().__init__(env, **kwargs)
        assert isinstance(self.env.observation_space, gym.spaces.Box) and len(self.env.observation_space.shape) == 1

        if low is not None:
            self.low = low
        if high is not None:
            self.high = high
        # Convert the environment to a discretized version
        self.bins = create_bins(self.low, self.high, feature_bins)

    def start_state(self):
        state, _ = self.env.reset(seed=int(self._reset_seed_gen.integers(0, 1e6)))
        return discretize(state, self.bins)

    def transition(self, action):
        next_state, reward, terminal, _, _ = self.env.step(action)
        reward = self.reward(next_state, reward)
        next_state = discretize(next_state, self.bins)
        return (next_state, reward, terminal)

############################################################

# Abstract class: an RlAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call get_action() to get an action, perform the action, and
# then provide feedback (via incorporate_feedback()) to the RL algorithm, so it can adjust
# its parameters.
class RlAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def get_action(self, state: state_t) -> action_t: raise NotImplementedError("Override me")

    # We will call this function when simulating an Mdp, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |next_state|.
    def incorporate_feedback(self, state: state_t, action: action_t, reward: int, next_state: state_t, terminal: bool):
        raise NotImplementedError("Override me")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.
class FixedRlAlgorithm(RlAlgorithm):
    def __init__(self, pi: Dict[state_t, action_t], actions: List[action_t], exploration_prob: float = 0.2):
        self.pi = pi
        self.actions = actions
        self.exploration_prob = exploration_prob

    # Just return the action given by the policy.
    def get_action(self, state: state_t, explore: bool = True) -> action_t:
        if explore and random.random() < self.exploration_prob:
            return random.choice(self.actions)
        else:
            return self.pi[state]

    # Don't do anything: just stare off into space.
    def incorporate_feedback(self, state: state_t, action: action_t, reward: int, next_state: state_t, terminal: bool): pass

# Class for untrained agent which takes random action every step.
# This class is used as a benchmark at the start of the assignment.
class RandomAgent(RlAlgorithm):
    def __init__(self, actions: List[action_t]):
        self.actions = actions

    def get_action(self, state: state_t, explore: bool = False):
        return random.choice(self.actions)

    def incorporate_feedback(self, state: state_t, action: action_t, reward: int, next_state: state_t, terminal: bool):
        pass

def polynomial_feature_extractor(
        state: state_t,
        degree: int = 3,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), degree 2, and scale [2, 1, 1], this should output:
    [1, 2x, y, z, 4x^2, y^2, z^2, 2xy, 2xz, yz, 4x^2y, 4x^2z, ..., 4x^2y^2z^2]
    '''
    if scale is None:
        scale = np.ones_like(state)

    # Create [1, s[0], s[0]^2, ..., s[0]^(degree)] array of shape (degree+1,)
    first_poly_feat = (state[0] * scale[0])**(np.arange(degree + 1))
    curr_poly_feat = first_poly_feat

    for i in range(1, len(state)):
        # Create [1, s[i], s[i]^2, ..., s[i]^(degree)] array of shape (degree+1,)
        new_poly_feat = (state[i] * scale[i])**(np.arange(degree + 1))

        # Do shape (len(curr_poly_feat), 1) times shape (1, degree+1) multiplication
        # to get broadcasted result of shape (len(curr_poly_feat), degree+1)
        # Note that this is also known as the vector outer product.
        curr_poly_feat = curr_poly_feat.reshape((len(curr_poly_feat), 1)) * new_poly_feat.reshape((1, degree + 1))

        # Flatten to (len(curr_poly_feat) * (degree+1),) array for the next iteration or final features.
        curr_poly_feat = curr_poly_feat.flatten()
    return curr_poly_feat


############################################################

# Perform |num_trials| of the following:
# On each trial, take the Mdp |mdp| and an RlAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the Mdp.
# Return the list of rewards that we get for each trial.
def simulate(mdp: Mdp, rl: RlAlgorithm, num_trials=10, train=True, verbose=False, demo=False):

    total_rewards = []  # The discounted rewards we get on each trial
    for trial in range(num_trials):
        state = mdp.start_state()
        if demo:
            mdp.env.render()
        total_discount = 1
        total_reward = 0
        trial_length = 0
        for _ in range(mdp.time_limit):
            if demo:
                time.sleep(0.05)
            action = rl.get_action(state, explore=train)
            if action is None:
                break
            next_state, reward, terminal = mdp.transition(action)
            trial_length += 1
            if train:
                rl.incorporate_feedback(state, action, reward, next_state, terminal)

            total_reward += total_discount * reward
            total_discount *= mdp.discount
            state = next_state

            if terminal:
                break # We have reached a terminal state

        if verbose and trial % 100 == 0:
            print(("Trial %d (total_reward = %s, Length = %s)" % (trial, total_reward, trial_length)))
        total_rewards.append(total_reward)
    return total_rewards


def sample_rl_trajectory(mdp: Mdp, rl: RlAlgorithm, train=True) -> List[Any]:
    traj = []
    state = mdp.start_state()

    while True:
        action = rl.get_action(state, explore=train)
        if action is None:
            break
        traj.append(action)
        next_state, reward, terminal = mdp.transition(action)
        if train:
            rl.incorporate_feedback(state, action, reward, next_state, terminal)
        state = next_state

        if terminal:
            break # We have reached a terminal state
    return traj
