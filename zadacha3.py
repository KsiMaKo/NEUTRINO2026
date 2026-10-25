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
    def get_intersection(self, pos, direction):
      """
      Чтобы фотон попадал на фотоумножитель, (\vec r0 + L \vec n)^2=R^2 
      т.е. точка куда прилетел должна лежать на сфере. Потом результат передаю в 
      find_fmt, чтобы определить, попало ли в детектор.
      """
        b = np.dot(pos, direction)
        c = np.dot(pos, pos) - self.R**2
        L = -b + np.sqrt(b**2 - c)
        return pos + L * direction

    def generate_fe(self):
        return rnd.poisson(self.a)

class Source:
    def __init__(self, Y=10000, pos=(0.0, 0.0, 0.0)):
        self.Y = Y 
        self.pos = np.array([0.0, 0.0, 0.0])
      
    def generate_directions(self, E):
        """
        Даёт случайные направления для фотонов энергии E.
        Возвращает массив единичных векторов (n_photons, 3)
        """
        n_photons = int(self.Y * E)
        phi = rnd.uniform(0, 2 * np.pi, n_photons)
        cos_theta = rnd.uniform(-1, 1, n_photons)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        vx = sin_theta * np.cos(phi)
        vy = sin_theta * np.sin(phi)
        vz = cos_theta
        
        return np.column_stack([vx, vy, vz])


det = Detector(R=2.0, N=40, r=0.1, a=0.5)
source = Source(Y=10000, pos=(0.0, 0.0, 0.0))
E_event = 1.0
directions = source.generate_directions(E=E_event)
n_photons = len(directions)

fmt_signals = np.zeros(det.N, dtype=int)
photons_count = 0

for direct in directions:
    pt = det.get_intersection(source.pos, direct)
    fmt_id = det.find_fmt(pt)
    if fmt_id != -1:
        photons_count += 1
        fe = det.generate_fe()
        fmt_signals[fmt_id] += fe

Q_total = np.sum(fmt_signals)
Q_ozhid = det.a * source.Y * E_event * (det.N * det.r**2) / (det.R**2)

print(f"Всего излучено фотонов: {n_photons}")
print(f"Попало фотонов: {photons_count}")
print(f"Полный собранный заряд Q: {Q_total}")
print(f"Оценка: {Q_ozhid:}")
