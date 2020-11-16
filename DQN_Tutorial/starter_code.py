# Import some modules from other libraries
import numpy as np
import torch
import time
from matplotlib import pyplot as plt
from replaybuffer import ReplayBuffer

# Import the environment module
from environment import Environment


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # map discrete to continuous actions
        self._action_map = {
            0: np.array([0, 0.1], dtype=np.float),
            1: np.array([0.1, 0], dtype=np.float),
            2: np.array([0, -0.1], dtype=np.float),
            3: np.array([-0.1, 0], dtype=np.float)
        }
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # return a random integer in [0,3] (inclusive) [N, E, S, W]
        return np.random.randint(0, 4)

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        return self._action_map[discrete_action]

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


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

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()
    
    # Train on batch of transitions
    def batch_train_q_network(self, batch):
        self.optimiser.zero_grad()
        loss = self._calculate_batch_loss(batch)
        loss.backward()
        self.optimiser.step()
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        state, discrete_action, reward, _ = transition
        s_tensor = torch.tensor([state]).float()
        a_tensor = torch.tensor([discrete_action])
        r_tensor = torch.tensor([[reward]])
        prediction = self.q_network.forward(s_tensor).gather(1, a_tensor.unsqueeze(1))
        return torch.nn.MSELoss()(prediction, r_tensor)
    
    def _calculate_batch_loss(self, batch):
        states = []
        actions = []
        rewards = []
        for state, action, reward, _ in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)

        s_tensor = torch.tensor(states).float()
        a_tensor = torch.tensor(actions)
        r_tensor = torch.tensor([rewards])

        prediction = self.q_network.forward(s_tensor).gather(1, a_tensor.unsqueeze(1))
        return torch.nn.MSELoss()(prediction, r_tensor)


# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()

    # Create a graph which will show the loss as a function of the number of training iterations
    fig, ax = plt.subplots()
    ax.set(xlabel='Episode', ylabel='Loss', title='Loss Curve for DQN with experience replay buffer')

    N_EPISODES = 100
    EPISODE_LENGTH = 20
    BUFFER_SIZE = 5000
    BATCH_SIZE = 100

    buffer = ReplayBuffer(BUFFER_SIZE)

    losses = np.zeros(N_EPISODES)
    # Loop over episodes
    for episode in range(N_EPISODES):

        episode_losses = np.zeros(EPISODE_LENGTH)
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(EPISODE_LENGTH):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()

            buffer.append(transition)

            if len(buffer) >= BATCH_SIZE:
                loss = dqn.batch_train_q_network(buffer.sample(BATCH_SIZE))
                episode_losses[step_num] = loss

            # time.sleep(0.2)
        
        losses[episode] = np.average(episode_losses)
        print("Finished episode {}, average loss = {}".format(episode, losses[episode]))

    # shift x-axis by BATCH_SIZE iterations
    ax.plot(losses, color='blue')
    plt.yscale('log')
    fig.savefig("dqn_erb_loss_vs_iterations.png")