import numpy as np
import random

def random_state(seed):
    # Convert seed to np.random.Randomstate
    
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def create_swiss_roll(n = 1000, rolls = 2.0, noise = 0.0, seed = None):
    """
    Generate a dataset which takes the form of a swiss roll.
    
    Parameters
    ----------
    n : int, default = 1000
        Number of data points
        
    rolls : float, default = 2.0
        Number of rolls in the swiss roll.
        
    noise : float, default = 0.0
        Standard deviation
        
    seed : int,
        Specific random number generator for creation of dataset, allowing 
        reproducible output. 
        
    Returns
    -------
    X : array of shape (n, 3)
        Generated datapoints. 
    """
    generator = random_state(None)
    
    t = rolls * np.pi * (1 + 2 * generator.rand(1, n))
    x = t * np.cos(t)
    y = 21 * generator.rand(1, n)
    z = t * np.sin(t)
    
    X = np.concatenate((x, y, z))
    X += noise * generator.randn(3, n)
    X = X.T
    t = np.squeeze(t)
    return X, t
