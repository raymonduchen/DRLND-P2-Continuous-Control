[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: ./Report.assets/ddpg.png
[image3]: ./Report.assets/architecture.png
[image4]: ./Report.assets/reward.png



## DRLND-P2 : Continuous Control Report

![Trained Agent][image1]

### Project introduction

The goal of this project is to train a deep reinforcement learning model that can control double-jointed arm (agent) to maintain its position at the target location for as many time steps as possible in a virtual environment [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) from [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). 

In this project, I trained a deep deterministic policy gradient ([DDPG](https://arxiv.org/abs/1509.02971)) model for the agent and it can get average score of +30 (over 100 consecutive episodes, and over all agents)

The implementation of DDPG algorithm is based on [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) provided from Udacity with some modifications such as hyperparameters tuning, multiple agents support, different neural network architecture, model update periodically and learning from experiences multiple times for each model update.

### DDPG Learning Algorithm

Deep deterministic policy gradient ([DDPG](https://arxiv.org/abs/1509.02971)) algorithm was introduced as an model-free off-policy actor-critic algorithm. It uses two deep neural networks as function approximators : actor and critic.

Actor neural network approximates policy  and critic neural network approximates action-value function. Actor policy determines action based on state and critic evaluates action-value based on state and action.

The detail DDPG algorithm are as follows :

![ddpg][image2]



#### Dealing with unstable learning

One of the biggest problem in reinforcement learning is its unstable learning. There're several techniques  used in this project DDPG implementation :

- **Experience replay using memory buffer**：Similar to DQN, DDPG also uses memory buffer to store several experience tuples with state, action, reward and next state `(s, a, r, s')`, and sample it randomly for learning to avoid sequential experience correlation.
- **Fixed target**：Similar to DQN, it's unstable to train model with moving target. Therefore, both DDPG actor and DDPG critic have their own pairs of local network and target network. 
- **Soft update**：Unlike DQN updates local network and target network by copying them for periodic steps, DDPG updates them gradually (softly) for every learning step that controlled by a soft update parameter.
- **Ornstein and Uhlenbeck Noise**：A major advantage of off policy algorithms such as DDPG is it can explore independently from the learning algorithm. Instead of using original actor policy, DDPG add it with noise sampled from Ornstein and Uhlenbeck noise. This could help a more explorative policy in training and find a better policy.
- **Gradient clipping**：To avoid gradient explosion, I added gradient clipping as suggested in Udacity course to clip the norm of the gradients at 1. It places a upper limit and avoids too large update to neural network parameters. 
- **Periodic and multiple learning**：Frequent learning with few experiences replay is not a good training strategy. In this project, Learning is made for every 20 timesteps and it learns 10 times for each learning by sampling from memory buffer randomly.



#### DDPG Neural Network Architecture 

Architecture of actor and critic neural networks are illustrated as follows : 

![architecture][image3]

The actor consists of 3 fully connected layer with 600, 400 and 4 units, respectively. The first two fully connected layers are followed with a ReLU nonlinear activation function and final fully connected layer is followed with tanh activation function. The actor takes state (size = 33) as input and output action (size=4).

| Actor        | Input size | Output size |
| ------------ | ---------- | ----------- |
| Layer1 (fc1) | 33         | 600         |
| ReLU         | 600        | 600         |
| Layer2 (fc2) | 600        | 400         |
| ReLU         | 400        | 400         |
| Layer3 (fc3) | 400        | 4           |
| Tanh         | 4          | 4           |

The critic consists of 3 fully connected layer with 600, 400 and 1 units, respectively. The first two fully connected layers are followed with a ReLU nonlinear activation function. The critic takes state (size = 33) and action (size = 4) as input and output action value (size=1). The first layer takes state as input (size = 33) and the second layer takes action and concatenate it with previous layer output as input (size = 600 + 4).

| Critic        | Input size | Output size |
| ------------- | ---------- | ----------- |
| Layer1 (fcs1) | 33         | 600         |
| ReLU          | 600        | 600         |
| Layer2 (fc2)  | 600 + 4    | 400         |
| ReLU          | 400        | 400         |
| Layer3 (fc3)  | 400        | 1           |



#### DDPG Hyperparameters

Key hyperparameters in this DDPG implementation are shown as follows : 

| Hyperparameter      | Value    | Meaning                                                 |
| ------------------- | -------- | ------------------------------------------------------- |
| n_episodes          | 300      | total max number of training episodes                   |
| max_t               | 1000     | total max number of steps in one episode                |
| BUFFER_SIZE         | int(1e6) | replay buffer size                                      |
| BATCH_SIZE          | 1024     | minibatch size sampled from replay buffer               |
| GAMMA               | 0.99     | discount factor                                         |
| TAU                 | 1e-3     | for soft update of target parameters                    |
| LR_ACTOR            | 1e-4     | learning rate of the actor                              |
| LR_CRITIC           | 1e-3     | learning rate of the critic                             |
| WEIGHT_DECAY        | 0        | L2 weight decay                                         |
| UPDATE_TIMESTEPS    | 20       | timestep period for updating model                      |
| MEMORY_SAMPLE_TIMES | 10       | times to learn from replay buffer for each model update |
| mu                  | 0        | the long-running mean in Ornstein-Uhlenbeck Noise       |
| theta               | 0.15     | the speed of mean reversion in Ornstein-Uhlenbeck Noise |
| sigma               | 0.2      | the volatility parameter in Ornstein-Uhlenbeck Noise    |



### Plot of Rewards

In this DDPG implementation, the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment is solved in ??? episodes with average score of +30 over 100 consecutive episodes, and over all agents.

The plot of reward in training stage is shown as follows :

![reward][image4]



### Future Work

- **Optimization hyperparameters and neural network architecture**：There're still lots of space we can try to adjust hyperparameters and neural network architecture. Different learning rate, replay buffer minibatch, different layers with different units or insertion with batch normalization layer may be a good try.
- **Another algorithms** : Aside from DDPG, there're lots of potential algorithms can be tested. For examples, [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization), [TRPO](https://arxiv.org/abs/1502.05477) (Trust Region Policy Optimization) or [D4PG](https://arxiv.org/abs/1804.08617) (Distributed Distributional Deterministic Policy Gradients) can be good candidates.
- **Priority replay buffer**：Replay buffer is equally and randomly sampled in this project. However, there're some experience tuples having high rewards that deserve high sampling probability. Implementing [priority experience replay](https://arxiv.org/abs/1511.05952) may help to improve total expected rewards.