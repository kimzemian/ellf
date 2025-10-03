#!/bin/bash


export WANDB_DIR=/share/dean/sweep/wandb
export WANDB_CACHE_DIR=/share/dean/wandb_cache

# Hyperparameter arrays
LRS=(1e-4 5e-4 1e-3 5e-3)
WDS=(1e-5 1e-4 1e-3 1e-2 1e-1)
OPTS=("adamw" "adam" "sgd")
WARMUPS=(0.05 0.1 1 0.5 1)
SCHEDULERS=("linear" "cosine" "constant_with_warmup" 
           "cosine_with_restarts --num_cycles 3"
           "cosine_with_restarts --num_cycles 5"
           "polynomial --power 5.0"
           "polynomial --power 0.5")

# Configuration
SCRIPT="main_bench.py"

# Run sweep sequentially
for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for opt in "${OPTS[@]}"; do
      for warmup in "${WARMUPS[@]}"; do
        for sched in "${SCHEDULERS[@]}"; do
          
          echo "Running: lr=$lr weight_decay=$wd optimizer=$opt warmup_epochs=$warmup scheduler=$sched" | tee -a sweep.log
          python $SCRIPT --lr $lr --weight_decay $wd --optimizer $opt --warmup_epoch $warmup --scheduler $sched --epochs 10 2>&1 | tee -a sweep.log
        done
      done
    done
  done
done

echo "Sweep complete!"
