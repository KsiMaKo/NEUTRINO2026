import time
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

#Without vectorisation
def simulate_event_tsykl(R, fmt_positions, r, a, lambda_zat, Y, E, pos):
    n_photons = int(Y * E)
    q_total = 0
    
    # 1. Генерация направлений по одному фотону в цикле
    directions = []
    for n in range(n_photons):
        phi = rnd.uniform(0, 2 * np.pi)
        cos_theta = rnd.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        directions.append([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    
    # 2. Пропагация каждого фотона по отдельности
    for d in directions:
        d = np.array(d)
        b = np.dot(pos, d)
        c = np.dot(pos, pos) - R**2
        L = -b + np.sqrt(b**2 - c)
        pt = pos + L * d
    
        # 3. Проверка попадания в каждый из 40 фотомножителей циклом
        hit = False
        for fmt_center in fmt_positions:
            diff = fmt_center - pt
            dist = np.sqrt(np.dot(diff, diff))
            if dist <= r:
                hit = True
                break
                
        # 4. Проверка затухания и генерация фотоэлектронов
        if hit:
            p_surv = np.exp(-L / lambda_zat)
            if rnd.random() < p_surv:
                q_total += rnd.poisson(a)   
    return q_total

#With vectorisation
class FastDetector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, lambda_zat=8.0):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.lambda_zat = lambda_zat
        self.fmt_positions = self.generate_fmts()

    def generate_fmts(self):
        phi = rnd.uniform(0, 2 * np.pi, self.N)
        cos_theta = rnd.uniform(-1, 1, self.N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        return np.column_stack([
            self.R * sin_theta * np.cos(phi),
            self.R * sin_theta * np.sin(phi),
            self.R * cos_theta
        ])

    def propagate_and_detect(self, pos, directions):
        # 1. Векторная пропагация фотонов до поверхности сферы
        b = np.dot(directions, pos)
        c = np.dot(pos, pos) - self.R**2
        L = -b + np.sqrt(b**2 - c)
        pts = pos + L[:, np.newaxis] * directions

        # 2. Векторная проверка попадания во все 40 фотомножителей
        diff = pts[:, np.newaxis, :] - self.fmt_positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        min_dist_sq = np.min(dist_sq, axis=1)
        hit_mask = min_dist_sq <= (self.r**2)

        # 3. Векторный учет затухания
        L_hits = L[hit_mask]
        p_surv = np.exp(-L_hits / self.lambda_zat)
        survived_mask = rnd.random(len(L_hits)) < p_surv
        n_detected = np.sum(survived_mask)

        # 4. Векторная генерация фотоэлектронов (Пуассон для всей суммы)
        if n_detected > 0:
            return rnd.poisson(n_detected * self.a)
        return 0


class FastSource:
    def __init__(self, Y=10000, pos=(0.0, 0.0, 0.0)):
        self.Y = Y
        self.pos = np.array(pos, dtype=float)

    def generate_photons(self, E):
        n_photons = rnd.poisson(self.Y * E)
        phi = rnd.uniform(0, 2 * np.pi, n_photons)
        cos_theta = rnd.uniform(-1, 1, n_photons)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        return np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

#Let's test them out agaisnt each other now
R = 2.0
N = 40
r = 0.1
a = 0.5
lambda_zat = 8.0
Y = 10000
E_event = 1.0
pos = np.array([0.0, 0.0, 0.0])

det_fast = FastDetector(R, N, r, a, lambda_zat)
src_fast = FastSource(Y, pos)

#To get rid of any inaccuracies due to FMTs positions, i won't regenerate them each time.
fmt_positions = det_fast.fmt_positions

n_test_events = 10

#Tsykl
t0 = time.time()
q_tsykl = []
for n in range(n_test_events):
    q = simulate_event_tsykl(R, fmt_positions, r, a, lambda_zat, Y, E_event, pos)
    q_tsykl.append(q)
time_tsykl = time.time() - t0
print(f"Время работы с циклами for: {time_tsykl:.3f}")

# numpy vectorisation
t0 = time.time()
q_numpy = []
for n in range(n_test_events):
    dirs = src_fast.generate_photons(E_event)
    q = det_fast.propagate_and_detect(src_fast.pos, dirs)
    q_numpy.append(q)
time_numpy = time.time() - t0
print(f"Время работы с numpy:{time_numpy:.3f} сек")

#Itogou
speedup = time_tsykl / time_numpy
print(f"С векторизацией выполнение быстрее в {speedup:.2f} раз")
print(f"Средний заряд с tsykl:  {np.mean(q_tsykl):.2f}")
print(f"Средний заряд с vectorisasion:  {np.mean(q_numpy):.2f}")
