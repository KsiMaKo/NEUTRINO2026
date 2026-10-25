import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

class Source:
    def __init__(self, Y=10000, pos=(0.0, 0.0, 0.0)):
        self.Y = Y
        self.pos = np.array(pos, dtype=float)

    def generate_photons(self, E, batch_size=100):
        """
        Генератор выдающий направления фотонов пачками по batch_size.
        """
        total_photons = rnd.poisson(self.Y * E)
        
        photons_left = total_photons
        while photons_left > 0:
            current_batch = min(batch_size, photons_left)
            
            phi = rnd.uniform(0, 2 * np.pi, current_batch)
            cos_theta = rnd.uniform(-1, 1, current_batch)
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            vx = sin_theta * np.cos(phi)
            vy = sin_theta * np.sin(phi)
            vz = cos_theta
            
            batch_directions = np.column_stack([vx, vy, vz])
            
            photons_left -= current_batch
        
            yield batch_directions
