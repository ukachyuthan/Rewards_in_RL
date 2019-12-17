import gym
import numpy as np
env = gym.make("FetchPickAndPlace-v1")
observation = env.reset(set_goal=[1.3,0.7,0.7],object_loc=[1.4,0.8])
# observation = env.reset()
val = 0.0
action = [0.0,0.0,0.0,0.01]
state = 0
count = 0

for _ in range(2000):
  env.render()


  # action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)


  # action[:-1] = observation['desired_goal'] - observation['observation'][0:3]
  # action[:-1] = (observation['observation'][3:6] - observation['observation'][0:3])

  if state == 0:
    action[:-1] = observation['observation'][6:9]*[1.0,1.0,0.9]
    # action = [0.0,0.0,1.0,0.0]
  elif state == 1:
    count+=1
    if count < 20:
      continue
    action[:-1] = observation['desired_goal'] - observation['observation'][0:3]
  else:
    action = [0.0,0.0,0.0,0.0]

  if done:
    print("done")
    # print(observation,reward, info,done)
    # observation = env.reset()
    action[3] = -0.01
    state = 1
    if info['is_success'] == 1.0:
      # observation = env.reset(set_goal=[1.3,0.7,0.7],object_loc=[1.4,0.8])
      observation = env.reset()
      state = 0
      action[3] = 0.1
      count = 0

    if np.mean(abs(observation['desired_goal'] - observation['observation'][0:3])) < 0.01:
      env.reset()
      state = 0
      action[3] = 0.1
      count = 0

env.close()