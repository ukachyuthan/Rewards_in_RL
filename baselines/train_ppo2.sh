source ./venv/bin/activate
python3.6 -W ignore -m baselines.run --alg=ppo2 --env=FetchPickAndPlace-v1 --num_timesteps=2.5e8 --save_path=./policies/ppo2a/fetchreach25kk --save_interval=100
