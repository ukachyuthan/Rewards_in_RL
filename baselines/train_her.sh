source ./venv/bin/activate
python3.6 -W ignore -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=2.5e6 --save_path=./policies/her_a/fetchreach25kk
