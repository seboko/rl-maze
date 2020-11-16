import numpy as np
import torch
import time
from matplotlib import pyplot as plt
from environment import Environment
from starter_code import Agent, Network, DQN

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
    ax.set(xlabel='Episode', ylabel='Loss', title='Loss Curve for DQN with online learning')

    N_EPISODES = 100
    EPISODE_LENGTH = 20

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

            loss = dqn.train_q_network(transition)
            # Sleep, so that you can observe the agent moving. Note: this line should be removed when you want to speed up training

            episode_losses[step_num] = loss
        
        losses[episode] = np.average(episode_losses)
        print("Finished episode {}, average loss = {}".format(episode, losses[episode]))

    ax.plot(losses, color='blue')
    plt.yscale('log')
    fig.savefig("dqn_online_loss_vs_episodes.png")