import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import cv2
from environment import Environment
from starter_code import Agent, Network, DQN
from replaybuffer import ReplayBuffer
from q_value_visualiser import QValueVisualiser

if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    # copy values from original dqn
    target = DQN()
    target.q_network.load_state_dict(dqn.q_network.state_dict())

    # # Create a graph which will show the loss as a function of the number of training iterations
    # fig, ax = plt.subplots()
    # ax.set(xlabel='Episode', ylabel='Loss', title='Loss Curve for DQN with experience replay buffer')

    N_EPISODES = 100
    EPISODE_LENGTH = 100
    BUFFER_SIZE = 5000
    BATCH_SIZE = 100
    TARGET_SWAP = 20

    buffer = ReplayBuffer(BUFFER_SIZE)

    losses = np.zeros(N_EPISODES)
    # Loop over episodes
    for episode in range(N_EPISODES):
        epsilon = min(10 / (episode + 1), 1)

        episode_losses = np.zeros(EPISODE_LENGTH)
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(EPISODE_LENGTH):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step(dqn, epsilon)

            buffer.append(transition)

            if len(buffer) >= BATCH_SIZE:
                loss = dqn.batch_train_q_network(buffer.sample(BATCH_SIZE), target_network=target)
                episode_losses[step_num] = loss

            if (episode * EPISODE_LENGTH + step_num) % TARGET_SWAP == 0:
                print("Swapped target network on step {}".format(episode * EPISODE_LENGTH + step_num))
                target.q_network.load_state_dict(dqn.q_network.state_dict())
            # time.sleep(0.05)
        
        losses[episode] = np.average(episode_losses)
        print("Finished episode {}, average loss = {}".format(episode, losses[episode]))

    
    # evaluate Q-value
    q_values = np.zeros((10, 10, 4))
    for col in range(10):
        x = col / 10 + 0.05
        for row in range(10):
            y = row / 10 + 0.05
            loc = torch.tensor([[x, y]], dtype=torch.float32)
            q_value = dqn.q_network.forward(loc)
            q_values[col, row] = q_value.detach().numpy()

    visualiser = QValueVisualiser(environment=environment, magnification=500)
    # Draw the image
    visualiser.draw_q_values(q_values, filename="q_values_epsilon_greedy.png")

    # draw greedy policy
    image = environment.draw(environment.init_state)
    loc = environment.init_state
    for _ in range(20):
        q = dqn.q_network.forward(torch.tensor(loc, dtype=torch.float32))
        greedy_direction = np.argmax(q.detach().numpy())
        action = agent._discrete_action_to_continuous(greedy_direction)
        next_loc, _ = environment.step(loc, action)
        loc_tuple = (int(loc[0] * environment.magnification),
                     int((1 - loc[1]) * environment.magnification))
        next_loc_tuple = (int(next_loc[0] * environment.magnification),
                          int((1 - next_loc[1]) * environment.magnification))

        cv2.line(image, loc_tuple, next_loc_tuple, (0,255,0), thickness=5)
        print("Location ", loc)
        loc = next_loc
    
    cv2.imwrite('epsilon_greedy_policy_returns.png', image)
        
