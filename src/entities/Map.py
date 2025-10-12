from dataclasses import dataclass

import numpy as np


@dataclass
class Map:
    """Class representing the map with rewards and distances."""
    
    rpositions: np.ndarray  # Reward positions (N x 2 array)
    rvalues: np.ndarray     # Reward values (N array)
    distmx: np.ndarray      # Distance matrix (N x N array)
