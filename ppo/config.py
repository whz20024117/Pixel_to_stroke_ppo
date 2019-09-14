config = {
    'ACTION_DIM': 14,
    'STATE_DIM': [28, 28],
    'MAX_ITER': 10000,
    'LR': 0.0001,
    'CLIP': 0.2,
    'GAMMA': 0.95,
    'C1': 1.0,      # value function parameter in loss
    'C2': 0.0,      # entropy bonus parameter in loss

    'MAX_STROKE': 20,
    'MAX_ITERATION': 10000,
    'T': 100,
    'TRAIN_EPOCH': 5,
    'BATCH_SIZE': 64,

    'SAVED_GAN': './saved_gan/1',
    'SAVED_POLICY': './saved_policy/1'
}