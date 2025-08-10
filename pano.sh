#!/bin/bash

# Generate panorama from text prompt
python3 demo_panogen_local.py \
--prompt "a beautiful street scene with buildings and trees" \
--output_path test_results/street \
--seed 42 \
--use_local