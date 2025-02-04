############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import time
import numpy as np
import torch
from collections import deque
import random
import matplotlib.pyplot as plt


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 3000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.discrete_action = None

        self.num_episodes = 0
        self.steps_in_episode = 0

        self.replaybuffer = ReplayBuffer(capacity=20000, epsilon=0.1, alpha=0.7)

        self.dqn = DQN()
        self.target = DQN()
        self.target.q_network.load_state_dict(self.dqn.q_network.state_dict())

        self.epsilon_init = 1
        self.epsilon = self.epsilon_init
        self.epsilon_decay = 0.1 ** (1 / 100)
        self.episode_len_decay = 0.8
        self.epsilon_min = 0.3
        self.gamma = 0.9
        self.batch_size = 1000
        self.target_swap = 200

        self._greedy = False
        self._found_greedy = False
        self._birthday = time.time()
        self._last_distance = 100 # This is only used for logging purposes
        self._found_goal_in_episode = False
        self._next_episode_length = self.episode_length


        # map discrete to continuous actions
        self._action_map = {
            0: np.array([0.02, 0], dtype=np.float),  # RIGHT
            1: np.array([0, 0.02], dtype=np.float),  # UP
            2: np.array([-0.02, 0], dtype=np.float), # LEFT
            3: np.array([0, -0.02], dtype=np.float)  # DOWN
        }

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        has_finished = self.steps_in_episode % self.episode_length == 0

        if has_finished:
            print(f"Finished episode {self.num_episodes} after {self.num_steps_taken} steps, episode_length={self.episode_length}, epsilon={self._current_epsilon()}, greedy={self._greedy}, last_distance={self._last_distance}")
            self.num_episodes += 1
            self.steps_in_episode = 0
            self._found_goal_in_episode = False
            self.epsilon *= self.epsilon_decay
            self.episode_length = self._next_episode_length

        return has_finished


    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        self.steps_in_episode += 1

        # try out the greedy policy every 5 episodes
        self._greedy = (self.num_episodes % 5 == 0 and self.steps_in_episode <= 100 and time.time() - self._birthday >= 480) or self._found_greedy
        
        if self._greedy:
            return self.get_greedy_action(state, False)

        discrete_action = self._choose_next_action(state)
        action = self._discrete_action_to_continuous(discrete_action)
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        self.discrete_action = discrete_action

        return action

    # Function for the agent to choose its next action
    def _choose_next_action(self, state):
        q = self.dqn.q_network.forward(torch.tensor([state]).float())
        best = torch.argmax(q).item()
        epsilon = self._current_epsilon()
        probs = np.full(4, epsilon / 4)
        probs[best] = 1 - epsilon + epsilon / 4
        return np.random.choice(range(4), p=probs)
    
    def _current_epsilon(self):
        return min(1, max(self.epsilon, self.epsilon_min))


    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        return self._action_map[discrete_action]

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):

        # check if we're on a greedy policy and have found the goal
        if self._greedy and self.steps_in_episode <= 100 and distance_to_goal < 0.03:
            self._found_greedy = True

        if distance_to_goal < 0.03 and not self._found_goal_in_episode:
            self._found_goal_in_episode = True
            self._next_episode_length = max(200, int(self.episode_length * self.episode_len_decay))
            print(f"Found goal, reducing next episode length to {self._next_episode_length}")
        
        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        if abs(next_state[0] - self.state[0]) < 0.0001 or abs(next_state[1] - self.state[1]) < 0.0001:
            reward -= 0.5
        
        # Only used for logging
        self._last_distance = distance_to_goal

        # Create a transition
        transition = (self.state, self.discrete_action, reward, next_state)

        self.replaybuffer.append(transition)

        # Only train when not trying greedy policy
        if not self._greedy:
            if len(self.replaybuffer) >= self.batch_size:
                batch = self.replaybuffer.sample(self.batch_size)
                self.dqn.batch_train_q_network(batch, self.gamma, self.target, self.replaybuffer)
        
            if self.num_steps_taken % self.target_swap == 0:
                self.target.q_network.load_state_dict(self.dqn.q_network.state_dict())





    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state, p=True):
        q = self.dqn.q_network.forward(torch.tensor([state]).float())
        if p:
            print("state {}, q {}".format(state, q))
        best = torch.argmax(q).item()
        return self._discrete_action_to_continuous(best)


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_3 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    
    # Train on batch of transitions
    def batch_train_q_network(self, batch, gamma, target_network, replaybuffer=None):
        self.optimiser.zero_grad()
        loss = self._calculate_batch_loss(batch, gamma, target_network, replaybuffer)
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def _calculate_batch_loss(self, batch, gamma, target_network, replaybuffer):
        states = batch[:,:2]
        actions = batch[:,2]
        rewards = batch[:,3]
        next_states = batch[:,4:]

        q_target = target_network.q_network.forward(torch.tensor(next_states).float())
        q_max_indices = torch.argmax(q_target, dim=1)

        q_max = self.q_network.forward(torch.tensor(next_states).float()).gather(1, q_max_indices.unsqueeze(1))
        r_tensor = torch.tensor(rewards).reshape((len(rewards), 1)) + gamma * q_max.reshape(len(rewards), 1)

        s_tensor = torch.tensor(states).float()
        a_tensor = torch.tensor(actions, dtype=torch.int64)

        prediction = self.q_network.forward(s_tensor).gather(1, a_tensor.unsqueeze(1))

        if replaybuffer is not None:
            replaybuffer.update_deltas(torch.abs(prediction - r_tensor.float()).detach().numpy())

        return torch.nn.MSELoss()(prediction, r_tensor.float())

class ReplayBuffer:
    def __init__(self, capacity=5000, epsilon=0.1, alpha=1):
        self._capacity = capacity
        self._epsilon = epsilon
        self._alpha = alpha
        self._buffer = np.zeros((capacity, 6), dtype=np.float)
        self._index = 0
        self._size = 0
        self._last_indices_returned = None
        self._weights = np.zeros(capacity)
        self._sampling_probs = np.zeros(capacity)
        self._can_append = True
    
    def append(self, transition):
        assert self._can_append
        state = [coord for coord in transition[0]]
        action = [transition[1]]
        reward = [transition[2]]
        next_state = [coord for coord in transition[3]]
        self._buffer[self._index] = np.array(state + action + reward + next_state)
        self._size = min(self._size + 1, self._capacity)

        self._weights[self._index] = np.max(self._weights)

        self._renormalize_weights()

        self._index = (self._index + 1) % self._capacity

    def update_deltas(self, deltas):
        self._can_append = True
        self._weights[self._last_indices_returned] = deltas.flatten()
        self._renormalize_weights()

    def _renormalize_weights(self):
        weights = (self._weights[:self._size] + self._epsilon) ** self._alpha
        self._sampling_probs[:self._size] = weights / np.sum(weights)

    def sample(self, n):
        self._can_append = False
        indices = np.random.choice(range(self._size), n, p=self._sampling_probs[:self._size])
        self._last_indices_returned = indices
        return self._buffer[indices]

    def __len__(self):
        return self._size