import numpy as np
import numpy.random as rnd
import time
import k3d

class BigDetector:
    def __init__(self, R=15.0, N=2000, r=0.1, a=0.5):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
      
        self.fmt_positions = self.generate_uniform_fmts()
        self.z_coords = self.fmt_positions[:, 2]

    def generate_uniform_fmts(self):
        i = np.arange(self.N, dtype=float)
        z = -self.R + (2.0 * self.R * (i + 0.5)) / self.N
        r_ring = np.sqrt(self.R**2 - z**2)
        angle = np.pi / np.e
        phi = i * angle
        
        x = r_ring * np.cos(phi)
        y = r_ring * np.sin(phi)
        
        return np.column_stack([x, y, z])

    def find_fmt_fast(self, point):
        """
        Проверяем только интервал [pz-r, pz+r].
        """
        pz = point[2]
        
        i_start = np.searchsorted(self.z_coords, pz - self.r)
        i_end = np.searchsorted(self.z_coords, pz + self.r)
      
        candidates = self.fmt_positions[i_start:i_end]
        if len(candidates) == 0:
          return -1
        diff = candidates - point
        dist_sq = np.sum(diff**2, axis=1)
        
        best = np.argmin(dist_sq)
        if dist_sq[best] <= self.r**2:
            return i_start + best
        return -1
    def find_fmt_slow(self, point):
        """
        Проверяем VSE
        """
        pz = point[2]
        
        i_start = np.searchsorted(self.z_coords, pz - self.r)
        i_end = np.searchsorted(self.z_coords, pz + self.r)
      
        candidates = self.fmt_positions[0:2000]
        diff = candidates - point
        dist_sq = np.sum(diff**2, axis=1)
        
        best = np.argmin(dist_sq)
        if dist_sq[best] <= self.r**2:
            return best
        return -1


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

det = BigDetector(R=15.0, N=2000, r=0.1, a=0.5)
source = Source(Y=10000)

dirs = source.generate_directions(E=1.0)
pts = det.R * dirs

t0 = time.time()
hits = 0
for p in pts:
    if det.find_fmt_fast(p) != -1:
        hits += 1
dt = time.time() - t0

t0 = time.time()
hits = 0
for p in pts:
    if det.find_fmt_slow(p) != -1:
        hits += 1
dts = time.time() - t0

print(f"10 000 фотонов быстро обработано за: {dt:.3f} с")
print(f"10 000 фотонов медленно обработано за: {dts:.3f} с")
print(f"Попало фотонов на катоды: {hits}")
print(f"Теоретическое ожидание: {int(10000 * (det.N * det.r**2) / (4 * det.R**2))}")

fmt_mesh = k3d.points(positions=det.fmt_positions.astype(np.float32), point_size=det.r, color=0x000100, shader='3d')
plot+= fmt_mesh
plot = k3d.plot(grid_visible=True, camera_auto_fit=True)
#Источник света
source_pt = k3d.points(
    positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32), point_size=0.5, color=0x0fff00, shader='3d')
plot +=fmt_mesh
plot +=source_pt
plot.display()
