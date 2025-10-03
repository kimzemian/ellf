# Scheduler configurations
SCHEDULER_CONFIGS = [
    # Basic schedulers
    {"name": "linear"},
    {"name": "cosine"},
    {"name": "constant_with_warmup"},
    # Schedulers with specific kwargs
    {"name": "cosine_with_restarts", "num_cycles": 3},
    {"name": "cosine_with_restarts", "num_cycles": 5},
    {"name": "polynomial", "power": 5.0},
    {"name": "polynomial", "power": 0.5},
]

# Hyperparameter grids
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3]
WEIGHT_DECAYS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
OPTIMIZERS = ["adamw", "adam", "sgd"]
WARMUP_EPOCHS = [0.05, 0.1, 1, 0.5, 1]
