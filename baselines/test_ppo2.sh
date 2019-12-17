source ./venv/bin/activate
python3.6 -m baselines.run --alg=ppo2 --env=FetchPickAndPlace-v1 --num_timesteps=0 --play --load_path=./policies/ppo2a/04400 --play
