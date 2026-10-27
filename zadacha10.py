import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pymc as pm
import pytensor.tensor as pt

rnd.seed(42)

R = 2.0
N_fmt = 40
r_fmt = 0.1
a = 0.5
Y = 10000.0
lambda_zat = 8.0

i = np.arange(N_fmt)
z_fmt = -R + 2.0 * R * (i + 0.5) / N_fmt
r_ring = np.sqrt(np.maximum(0.0, R**2 - z_fmt**2))
phi_fmt = i * np.pi / np.e
x_fmt = r_ring * np.cos(phi_fmt)
y_fmt = r_ring * np.sin(phi_fmt)
fmt_pos = np.column_stack([x_fmt, y_fmt, z_fmt])

E_true = 2.718
pos_true = np.array([0.27, 0.10, 0.26])


def get_expected_mu(pos, E):

    diff = fmt_pos - pos
    dist = np.sqrt(np.sum(diff**2, axis=1))
  
    cos_theta = np.sum(diff * fmt_pos, axis=1) / (dist * R)

    p_geom = (r_fmt**2 * cos_theta) / (4.0 * dist**2)
    p_zat = np.exp(-dist / lambda_zat)

    return E * Y * a * p_geom * p_zat

mu_true = get_expected_mu(pos_true, E_true)
q_real = rnd.poisson(mu_true)

print(f"Сгенерирован сэмпл отклика для {N_fmt} ФЭУ:")
print(f"Суммарный заряд: {np.sum(q_real)} ф.э. (ожидалось: {np.sum(mu_true):.1f})")
print(f"Диапазон заряда на каналах: min = {np.min(q_real)}, max = {np.max(q_real)}")

with pm.Model() as recon_model:
    E = pm.Uniform("E", lower=0.01, upper=10.0)
    x = pm.Uniform("x", lower=-1, upper=1)
    y = pm.Uniform("y", lower=-1, upper=1)
    z = pm.Uniform("z", lower=-1, upper=1)

    pos = pt.stack([x, y, z])

    diff = fmt_pos - pos
    dist = pt.sqrt(pt.sum(diff**2, axis=-1))

    dot_prod = pt.sum(diff * fmt_pos, axis=-1)
    cos_theta = dot_prod / (dist * R)
  
    p_geom = (r_fmt**2 * cos_theta) / (4.0 * dist**2)
    p_zat = pt.exp(-dist / lambda_zat)

    # Теоретическое среднее для каждого канала
    mu = E * Y * a * p_geom * p_zat

    # Likelihood с наблюдаемыми данными q_real
    obs = pm.Poisson("obs", mu=mu, observed=q_real)

    # Сэмплирование апостериорного распределения
    idata = pm.sample(draws=2500, tune=1000, chains=2, target_accept=0.9, random_seed=42, progressbar=True)

#Теперь обратноищем

post_E = idata.posterior["E"].values.flatten()
post_x = idata.posterior["x"].values.flatten()
post_y = idata.posterior["y"].values.flatten()
post_z = idata.posterior["z"].values.flatten()

rec_E_mean, rec_E_std = np.mean(post_E), np.std(post_E)
rec_x_mean, rec_x_std = np.mean(post_x), np.std(post_x)
rec_y_mean, rec_y_std = np.mean(post_y), np.std(post_y)
rec_z_mean, rec_z_std = np.mean(post_z), np.std(post_z)

sigma_pos = np.sqrt(rec_x_std**2 + rec_y_std**2 + rec_z_std**2)

print(f"{'Параметр'} | {'Истинное'} | {'Реконструкция'}")
print(f"{'E (МэВ)'} | {E_true:.3f} | {rec_E_mean:.3f} ± {rec_E_std:.3f}")
print(f"{'x (м)'} | {pos_true[0]:.3f} | {rec_x_mean:.3f} ± {rec_x_std:.3f}")
print(f"{'y (м)'} | {pos_true[1]:.3f} | {rec_y_mean:.3f} ± {rec_y_std:.3f}")
print(f"{'z (м)'} | {pos_true[2]:.3f} | {rec_z_mean:.3f} ± {rec_z_std:.3f}")

print(f"Пространственное отклонение: {sigma_pos:.3f}")
