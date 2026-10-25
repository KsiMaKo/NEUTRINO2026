import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.fmt_positions = self.generate_fmts()

    def generate_fmts(self):
        phi = rnd.uniform(0, 2 * np.pi, self.N)
        cos_theta = rnd.uniform(-1, 1, self.N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        x = self.R * sin_theta * np.cos(phi)
        y = self.R * sin_theta * np.sin(phi)
        z = self.R * cos_theta
        return np.column_stack([x, y, z])

    def count_hits(self, pos, directions):
        """
        Переписал немного относительно предыдущих задач более эффективно. 
        directions-направления фотонов
        """
        b = np.dot(directions, pos)
        c = np.dot(pos, pos) - self.R**2
        L = -b + np.sqrt(b**2 - c)
        pts = pos + L[:, np.newaxis] * directions

        diff = pts[:, np.newaxis, :] - self.fmt_positions[np.newaxis, :, :]#r<<R, поэтому мне этого достаточно, чтобы определить попадание
        dist_sq = np.sum(diff**2, axis=-1)
        
        min_dist_sq = np.min(dist_sq, axis=1)
        hits = np.sum(min_dist_sq <= self.r**2)
        return hits

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


def simulate_event(det, source, E):
    dirs = source.generate_directions(E)
    hits = det.count_hits(source.pos, dirs)
    if hits > 0:
        Q = rnd.poisson(hits * det.a)
    else:
        Q = 0
    return Q


det = Detector(R=2.0, N=40, r=0.1, a=0.5)
source_center = Source(Y=10000, pos=(0.0, 0.0, 0.0))

N_events = 1000
Q_events_1MeV = np.zeros(N_events)

for i in range(N_events):
    Q_events_1MeV[i] = simulate_event(det, source_center, E=1.0)

mean_Q = np.mean(Q_events_1MeV)
std_Q = np.std(Q_events_1MeV)

print(f"Средний заряд <Q>: {mean_Q:}")
print(f"Стандартное отклонение sigma_Q: {std_Q:}")
print(f"Относительная ширина sigma_Q / <Q>: {std_Q / mean_Q:}")

plt.figure(figsize=(7, 4))
plt.hist(Q_events_1MeV, bins=25, density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Распределение собранного заряда (1000 событий, 1 МэВ)\n<Q> = {mean_Q:.1f}, $\sigma$ = {std_Q:.1f}")
plt.xlabel("Полный заряд Q")
plt.ylabel("Плотность вероятности")
plt.grid(alpha=0.3)
plt.show()


energies = [0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
rel_widths = []
mean_charges = []

for E in energies:
    q_list = np.zeros(N_events)
    for i in range(N_events):
        q_list[i] = simulate_event(det, source_center, E)
    
    m = np.mean(q_list)
    s = np.std(q_list)
    mean_charges.append(m)
    rel_widths.append(s / m)
    print(f"E = {E:} МэВ:  <Q> = {m:} ф.э.,  sigma/Q = {s/m:}")

plt.figure(figsize=(7, 4))
plt.plot(energies, rel_widths, 'o-', color='skyblue')
plt.title(r'Зависимость относительной ширины $\sigma_Q / \langle Q \rangle$ от энергии')
plt.xlabel('Энергия E, МэВ')
plt.ylabel(r'$\sigma_Q / \langle Q \rangle$')
plt.grid(alpha=0.3)
plt.show()



source_shifted = Source(Y=10000, pos=(0.0, 0.0, 1.0))

Q_shifted = np.zeros(N_events)
for i in range(N_events):
    Q_shifted[i] = simulate_event(det, source_shifted, E=1.0)

mean_Q_center = np.mean(Q_events_1MeV)
mean_Q_shift = np.mean(Q_shifted)
diffq = (mean_Q_shift - mean_Q_center) / mean_Q_center

print(f"Средний заряд в центре (z = 0): {mean_Q_center:}")
print(f"Средний заряд со сдвигом (z = 1): {mean_Q_shift:}")
print(f"Изменение среднего заряда: {diffq:}")
