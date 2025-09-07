#!/bin/bash

./docker/cluster/submit_job_neuronic.sh 1_1 /n/fs/pvl-procur/anybody scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_tf_reach.yaml RUN_SEED 23
./docker/cluster/submit_job_neuronic.sh 1_1 /n/fs/pvl-procur/anybody scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_tf_reach.yaml RUN_SEED 34
./docker/cluster/submit_job_neuronic.sh 1_1 /n/fs/pvl-procur/anybody scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_tf_reach.yaml RUN_SEED 42
./docker/cluster/submit_job_neuronic.sh 1_2 /n/fs/pvl-procur/anybody scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml RUN_SEED 23
./docker/cluster/submit_job_neuronic.sh 1_2 /n/fs/pvl-procur/anybody scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml RUN_SEED 34
./docker/cluster/submit_job_neuronic.sh 1_2 /n/fs/pvl-procur/anybody scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml RUN_SEED 42
