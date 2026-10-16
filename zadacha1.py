import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import k3d

class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5):
        self.R = R    
        self.N = N    
        self.r = r     
        self.a = a      
        
        self.fmt_positions = self.generate_fmts()

    def generate_fmts(self):
        """
        Равномерно распределяю N центров ФЭУ по сфере радиуса R
        """
        phi = rnd.uniform(0, 2 * np.pi, self.N)
        cos_theta = rnd.uniform(-1, 1, self.N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        x = self.R * sin_theta * np.cos(phi)
        y = self.R * sin_theta * np.sin(phi)
        z = self.R * cos_theta
        
        return np.column_stack([x, y, z])

    def find_fmt(self, point):
        """
        Проверяю, попадает ли точка на ФЭУ.
        Возвращает номер ФЭУ (0..N-1) или -1, если мимо.
        """
        diff = self.fmt_positions - point
        dist = np.sqrt(np.sum(diff**2, axis=1))
        
        best_fmt = np.argmin(dist)
        if dist[best_fmt] <= self.r:
            return int(best_fmt)
        return -1
        
det = Detector(R=2.0, N=40, r=0.1, a=0.5)
plot = k3d.plot(name="Детектор")

fmt_mesh = k3d.points(positions=det.fmt_positions.astype(np.float32), point_size=det.r, color=0x000000)
plot += fmt_mesh

plot.display()
