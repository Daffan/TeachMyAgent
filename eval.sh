#!/bin/bash

python TeachMyAgent/students/test_policy_latent_space.py \
	--fpath TeachMyAgent/students/data/alp_gmm_${1}/alp_gmm_${1}_s16 \
	-nr --fixed_test_set parametric_stumps_test_set \
	--env parametric-continuous-stump-tracks-v0 \
	--max_stump_h 3.0 --min_stump_h 0 --max_obstacle_spacing 6.0 \
	--embodiment old_classic_bipedal --student ppo --lr 0.0003 --backend tf1 \
	--steps_per_ep 500000 --nb_test_episode 100 --nb_env_steps 20 --checkpoint 01000 \
	--episode_ids -1
