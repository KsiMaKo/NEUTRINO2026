import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, lambda_zat=8.0):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.lambda_zat = lambda_zat #длина затухания
        self.fmt_positions = self.generate_fmts()

    def generate_fmts(self):
        phi = rnd.uniform(0, 2 * np.pi, self.N)
        cos_theta = rnd.uniform(-1, 1, self.N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        x = self.R * sin_theta * np.cos(phi)
        y = self.R * sin_theta * np.sin(phi)
        z = self.R * cos_theta
        return np.column_stack([x, y, z])

    def count_hits(self, pos, directions, zatuhanie=True):
        """
        Считает число фотонов, попавших в катоды с учетом затухания.
        """
        b = np.dot(directions, pos)
        c = np.dot(pos, pos) - self.R**2
        L = -b + np.sqrt(b**2 - c)
        pts = pos + L[:, np.newaxis] * directions

        diff = pts[:, np.newaxis, :] - self.fmt_positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        min_dist_sq = np.min(dist_sq, axis=1)
        hit_mask = (min_dist_sq <= self.r**2)

        if not zatuhanie:
            return np.sum(hit_mask)
          
        L_hits = L[hit_mask]
        p_surv = np.exp(-L_hits / self.lambda_zat)
        survived = rnd.random(len(L_hits)) < p_surv
        return np.sum(survived)

class Source:
    def __init__(self, Y=10000, pos=(0.0, 0.0, 0.0)):
        self.Y = Y
        self.pos = np.array(pos, dtype=float)

    def generate_directions(self, E):
        n_photons = int(self.Y * E)
        phi = rnd.uniform(0, 2 * np.pi, n_photons)
        cos_theta = rnd.uniform(-1, 1, n_photons)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        vx = sin_theta * np.cos(phi)
        vy = sin_theta * np.sin(phi)
        vz = cos_theta
        return np.column_stack([vx, vy, vz])


def simulate_event(det, source, E=1.0, zatuhanie=True):
    dirs = source.generate_directions(E)
    hits = det.count_hits(source.pos, dirs, zatuhanie=zatuhanie)
    if hits > 0:
        return rnd.poisson(hits * det.a)
    return 0

#Теперь моделируем
det = Detector(R=2.0, N=40, r=0.1, a=0.5, lambda_zat=8.0)
z_coords = [0.0, 0.5, 1.0, 1.5]
N_events = 1000
E_event = 1.0

print("Запуск симуляции")
for z in z_coords:
    src = Source(Y=10000, pos=(0.0, 0.0, z))
    
    # 1. Без затухания
    q_no = np.array([simulate_event(det, src, E=E_event, zatuhanie=False) for n in range(N_events)])
    
    # 2. С затуханием
    q_zat = np.array([simulate_event(det, src, E=E_event, zatuhanie=True) for n in range(N_events)])
        
    print(f"z = {z} м | Без затухания: <Q> = {np.mean(q_no):.3f}±{np.std(q_no):.3f} | С затуханием: <Q> = {np.mean(q_zat):.3f}±{np.std(q_zat):.3f}")
