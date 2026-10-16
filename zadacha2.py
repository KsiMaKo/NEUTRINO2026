import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import k3d

class Source:
    def __init__(self, Y=10000):
        self.Y = Y 
        self.pos = np.array([0.0, 0.0, 0.0])
      
    def generate_directions(self, E):
        """
        Разыгрывает случайные изотропные направления фотонов для энергии E (МэВ).
        Возвращает массив единичных векторов размера (N_photons, 3).
        """
        n_photons = int(self.Y * E)
        phi = rnd.uniform(0, 2 * np.pi, n_photons)
        cos_theta = rnd.uniform(-1, 1, n_photons)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        vx = sin_theta * np.cos(phi)
        vy = sin_theta * np.sin(phi)
        vz = cos_theta
        
        return np.column_stack([vx, vy, vz])
