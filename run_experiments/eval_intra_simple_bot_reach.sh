#!/bin/bash

./docker/cluster/submit_job_ionic.sh test4 /n/fs/pvl-exptrack/anybody scripts/run.py --headless --enable_cameras OVERRIDE_CFGNAME experiment_cfgs/eval_mt.yaml EVAL_CHECKPOINT LOGS_PATH/intra_simple_bot_reach/Tr0-c0-s0-t-nt-h0_42/checkpoints/agent_100000.pt
