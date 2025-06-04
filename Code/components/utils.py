import torch
import matplotlib.pyplot as plt
import os

def plot_durations(title, episode_rewards, running_average_rewards, path):
    # If interactive mode is not on, turn it on
    if not plt.get_fignums():
        plt.ion()  # Turn on interactive mode

    # Clear the previous plot and prepare a new one
    plt.clf()

    episode_rewards_tensor = torch.FloatTensor(episode_rewards)
    running_average_rewards_tensor = torch.FloatTensor(running_average_rewards)

    # Plot the graph
    plt.plot(episode_rewards_tensor.numpy(), color='b', label=f'Episode Reward')
    plt.plot(running_average_rewards_tensor.numpy(), color='r', label=f'Running Reward')

    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()

    RunTime = len(episode_rewards_tensor)
    if len(episode_rewards_tensor) % 200 == 0:
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + 'RunTime_' + str(RunTime) + '_graph.jpg')

    # Update the figure without opening a new window
    plt.draw()
    plt.pause(0.0001)  # This pauses to update the figure

# Optional: When finished plotting, turn off interactive mode if needed
# plt.ioff()  # Uncomment to stop interactive mode
