# Reward Shaping

In this project my team and I sought to study the impact of engineering different rewards for the 'FetchPickAndPlace-v1' environment from OpenAI's gym.

We have used the baselines provided by OpenAI as the infrastructure to train the model.

The models were trained using DDPG with sparse rewards negated by Hindsight Experience replay. A more in-depth explanation is provided in [Link](https://openai.com/blog/ingredients-for-robotics-research/)

We tried to work on the intuition that making the reward function more intuitive would improve the training performances. We thus tried to innovate upon the reward function provided by OpenAI to see if we could improve training performances. We also sought to understand if reward functions could also influence how the models learned to execute certain tasks.