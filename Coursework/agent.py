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

import numpy as np
import torch
from collections import deque
import random


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 1000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.discrete_action = None

        self.replaybuffer = ReplayBuffer()

        self.dqn = DQN()
        self.target = DQN()
        self.target.q_network.load_state_dict(self.dqn.q_network.state_dict())

        self.epsilon_init = 1
        self.epsilon_decay = 0.1 ** (1 / 20)
        self.epsilon_min = 0.1
        self.gamma = 0.9
        self.batch_size = 1000
        self.target_swap = 20

        # map discrete to continuous actions
        self._action_map = {
            0: np.array([0.02, 0], dtype=np.float),  # RIGHT
            1: np.array([0, 0.02], dtype=np.float),  # UP
            2: np.array([-0.02, 0], dtype=np.float), # LEFT
            3: np.array([0, -0.02], dtype=np.float)  # DOWN
        }

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        discrete_action = self._choose_next_action(state)
        action = self._discrete_action_to_continuous(discrete_action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
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
        ep = max(self.epsilon_init * self.epsilon_decay ** (self.num_steps_taken // self.episode_length), self.epsilon_min)
        if self.num_steps_taken % self.episode_length == 0:
            print("episode={}, steps={}, epsilon={}".format(self.num_steps_taken // self.episode_length, self.num_steps_taken, ep))
        return ep


    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        return self._action_map[discrete_action]

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        # Create a transition
        transition = (self.state, self.discrete_action, reward, next_state)

        self.replaybuffer.append(transition)

        if len(self.replaybuffer) >= self.batch_size:
            batch = self.replaybuffer.sample(self.batch_size)
            self.dqn.batch_train_q_network(batch, self.gamma, self.target)
        
        if self.num_steps_taken % self.target_swap == 0:
            self.target.q_network.load_state_dict(self.dqn.q_network.state_dict())





    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        q = self.dqn.q_network.forward(torch.tensor([state]).float())
        best = torch.argmax(q).item()
        print("state = {}, best_action={}".format(state, best))
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
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    
    def state_dict_eq(self, d1, d2):
        for (k1, v1), (k2, v2) in zip(d1.items(), d2.items()):
            if k1 != k2:
                print("keys not equal", k1, k2)
                return False
            if not torch.equal(v1, v2):
                print("tensors not equal")
                return False
        return True

    # Train on batch of transitions
    def batch_train_q_network(self, batch, gamma=0.9, target_network=None, replaybuffer=None):
        if target_network is not None:
            target_net_weights = target_network.q_network.state_dict()
        self.optimiser.zero_grad()
        loss = self._calculate_batch_loss(batch, gamma, target_network, replaybuffer)
        loss.backward()
        self.optimiser.step()
        if target_network is not None:
            assert self.state_dict_eq(target_network.q_network.state_dict(), target_net_weights)
        return loss.item()

    def _calculate_batch_loss(self, batch, gamma, target_network, replaybuffer):
        states = []
        actions = []
        rewards = []
        next_states = []
        for state, action, reward, next_state in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

        q = (target_network.q_network.forward(torch.tensor(next_states).float()) if target_network is not None
             else self.q_network.forward(torch.tensor(next_states).float()))
        q_max, _ = torch.max(q, 1)
        r_tensor = torch.tensor(rewards).reshape((len(rewards), 1)) + gamma * q_max.reshape(len(rewards), 1)

        s_tensor = torch.tensor(states).float()
        a_tensor = torch.tensor(actions)

        # if replaybuffer is not None:

        prediction = self.q_network.forward(s_tensor).gather(1, a_tensor.unsqueeze(1))
        return torch.nn.MSELoss()(prediction, r_tensor.float())

class ReplayBuffer:
    def __init__(self, size=5000):
        self._buffer = deque(maxlen=size)
    
    def append(self, transition):
        self._buffer.append(transition)
        return self
    
    def sample(self, n):
        return random.choices(self._buffer, k=n)
    
    def __len__(self):
        return len(self._buffer)