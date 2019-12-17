source ./venv/bin/activate
python3.6 -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=0 --load_path=./policies/her_a/fetchreach25kk --play
