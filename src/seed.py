import torch
import random
import numpy as np

def set_seed(seed):
    # --- Global seeds ---
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # --- For CUDA (if using) ---
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- Ensure deterministic behavior (optional, slower) ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- DataLoader shuffle ---
    g = torch.Generator()
    g.manual_seed(seed)

    return g
