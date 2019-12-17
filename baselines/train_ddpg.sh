source ./venv/bin/activate
python3.6 -W ignore -m baselines.run --alg=ddpg --env=FetchPickAndPlace-v1 --num_timesteps=2.5e8 --save_path=./policies/ddpg/fetchreach25kk --save_interval=100
