[//]: # "Image References"

[image1]: ./Report.assets/agent_tested.gif "Trained Agent"

# DRLND-P2 : Continuous Control

![Trained Agent][image1]

### Project Details

This is my work for the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) (DRLND) Project 2 : Continuous Control. The goal of this project is to train a deep reinforcement learning model that can control double-jointed arm (agent) to maintain its position at the target location for as many time steps as possible in a virtual environment [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) from [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). 

In this project, I trained a deep deterministic policy gradient ([DDPG](https://arxiv.org/abs/1509.02971)) model for the agent and it can get average score of +30 (over 100 consecutive episodes, and over all agents)

**Environment**

The virtual environment [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) used in this project is from [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents) (version v0.4). There're two separate versions of the Unity environment.

- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

**Reward**

A reward of **+0.1** is provided for each step that the agent's hand is in the goal location.

**Observation space**

The observation space consists of **33** variables corresponding to position, rotation, velocity, and angular velocities of the arm.

**Action space**

Each action is a vector with **4** numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between **-1** and **1**.

**Goal**

The goal of your agent is to maintain its position at the target location for as many time steps as possible.



### Solving the Environment

The project need only solve one of the two versions of the environment as follows. The second version is solved in this project.

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 



### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the GitHub repository, in the `DRLND-P2-Continuous-Control/` folder, and unzip (or decompress) the file. 



### Instructions

#### Train agent

1. Activate the conda environment `drlnd` as established in [Udacity deep reinforcement learning repository](https://github.com/udacity/deep-reinforcement-learning)

2. Open jupyter notebook `Continuous_Control.ipynb`

3. Change kernel to `drlnd` in `Continuous_Control.ipynb`

4. Change environment path in the first code cell, e.g.

   ```
   env = UnityEnvironment('Reacher20_Linux/Reacher.x86_64')   # Linux
   ```

5. Execute code cells in `Continuous_Control.ipynb`, and trained model will be saved in `solved_checkpoint_actor.pth` and `solved_checkpoint_critic.pth` as the average score > +30.0.

#### Test agent

1. Activate the conda environment `drlnd` as established in [Udacity deep reinforcement learning repository](https://github.com/udacity/deep-reinforcement-learning)

2. Open jupyter notebook `Continuous_Control_Test.ipynb`

3. Change kernel to `drlnd` in `Continuous_Control_Test.ipynb`

4. Change environment path in the first code cell, e.g.

   ```
   env = UnityEnvironment('Reacher20_Linux/Reacher.x86_64')   # Linux
   ```

5. Execute code cells in `Continuous_Control_Test.ipynb`, and the trained model will be tested for 3 times.

#### Note

As a default, twenty agents version is trained in this project, if you would like to train and test using one agent version, only change the environment path from 20 agents' environment to 1 agent's environment in `Continuous_Control.ipynb` and `Continuous_Control_Test.ipynb`, e.g. change the environment path from

```
env = UnityEnvironment('Reacher20_Linux/Reacher.x86_64')   # 20 agent env path (Linux)
```

to

```
env = UnityEnvironment('Reacher1_Linux/Reacher.x86_64')   # 1 agent env path (Linux)
```

