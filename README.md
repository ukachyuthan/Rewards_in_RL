# Reward Shaping

In this project my team and I sought to study the impact of engineering different rewards for the 'FetchPickAndPlace-v1' environment from OpenAI's [gym.](https://github.com/openai/gym/tree/master/gym/envs/robotics)

We have used the baselines provided by OpenAI as the infrastructure to train the model.

The models were trained using DDPG with sparse rewards negated by Hindsight Experience replay. A more in-depth explanation is provided in [here.](https://openai.com/blog/ingredients-for-robotics-research/)

We tried to work on the intuition that making the reward function more intuitive would improve the training performances. We thus tried to innovate upon the reward function provided by OpenAI to see if we could improve training performances. We also sought to understand if reward functions could also influence how the models learned to execute certain tasks.

# Procedure

The reward functions we created work on the fact that reducing distances along the goal x, y and z coordinates will lead to better results. So we added additional penalties based on the gripper's distance from these axes.

We also additionally worked on a slight modification of the above mentioned reward function to train the model to move the gripper in a Manhattan/Taxi Cab trajectory. Below is an image describing this trajectory from [Wikipedia.](https://en.wikipedia.org/wiki/Taxicab_geometry)

![Image](https://github.com/ukachyuthan/Rewards_in_RL/blob/master/Manhattan_distance.svg)

