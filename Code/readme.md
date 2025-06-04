# Reinforcement Learning Project: CartPole-v1 and LunarLander-v3

This project implements Reinforcement Learning algorithms for solving the CartPole-v1 and LunarLander-v3 environments using Deterministic and Stochastic modes, with an optional Hierarchical Reinforcement Learning (HRL) agent.

---

## Requirements

Before running the code, ensure you have all the required dependencies installed. Use the following command:

pip install -r requirements.txt


Running the Code

For CartPole-v1:
Navigate to the folder CartPole-v1
Run cart_pole_main.py


For LunarLander-v3:
Navigate to the folder LunarLander-v3
Run lunar_lander_main.py



Input Prompts
The scripts will prompt for the following inputs:

Seed Value
To ensure repeatability, you can input a seed value when prompted:
Enter a seed value (e.g., 42):


Mode Selection
You can run the environment in either Deterministic or Stochastic mode.
Choose a mode when prompted:
Choose mode: Deterministic (D) or Stochastic (S):


HRL Agent
You can enable the usage of the Hierarchical Reinforcement Learning agent when prompted:
Do you want to use the HRL agent? (Y/N):


Upon successful input of all data, the code will start running with a live graph opened in a window. 
The terminal will constantly provide update as such:
"Episode xxxx : Steps = xxxx, Total Reward = xxx, Running Average Reward = xxx"


When the enviroment is considered solved, this message will appear in your terminal ( XX is their respective number ):
"Environment solved after xxx episodes with average reward xxx over the last 100 episodes"


Results and Outputs
The results and plots generated during the experiments are stored in the Output directory, organized by environment and experiment type.
Output
├── CartPole-v1
│   ├── HRL_NoisyDueling_DQN_CartPole-v1_Deterministic
│   ├── HRL_NoisyDueling_DQN_CartPole-v1_Stochastic
│   ├── NoisyDueling_DQN_CartPole-v1_Deterministic
│   └── NoisyDueling_DQN_CartPole-v1_Stochastic
├── LunarLander-v3
│   ├── HRL_NoisyDueling_DQN_LunarLander-v3_Deterministic
│   ├── HRL_NoisyDueling_DQN_LunarLander-v3_Stochastic
│   ├── NoisyDueling_DQN_LunarLander-v3_Deterministic
│   └── NoisyDueling_DQN_LunarLander-v3_Stochastic
