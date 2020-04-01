from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.IN_PLANES = 2
_C.MODEL.OUT_PLANES = 1

_C.MODEL.KERNERLS = 8
_C.MODEL.POOLING = 'max_pool'
_C.MODEL.BIAS = True

_C.MODEL.RESIDUAL = False
_C.MODEL.NON_LOCAL = True

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN()
_C.DATA_LOADER.SHUFFLE = True
_C.DATA_LOADER.NUM_WORKERS = 1

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

# Total number of training epoches
_C.SOLVER.TOTAL_EPOCH = 100

_C.SOLVER.BASE_LR = 0.01

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 5e-4

# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

# Learning rate and weight decay for the bias
# can be different from the base LR and decay.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.GAMMA = 0.1

_C.SOLVER.SAMPLES_PER_BATCH = 16

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()