import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pymc as pm

R = 2.0
N_fmt = 40
r = 0.1
a_eff = 0.5
Y_yield = 10000
E_event = 1.0
lambda_zat = 8.0

# Геометрическая вероятность попадания фотона из центра в фотомножитель
p_geom = (N_fmt * r**2) / (4.0 * R**2)

# Вероятность пропагации фотона с учетом поглощения
p_zat = np.exp(-R / lambda_zat)

print(f"Геометрическая вероятность: {p_geom:.2f}")
print(f"Вероятность выживания на L: {p_zat:.2f}")

#Без затухания
with pm.Model() as model0:
    N0_photons = E_event * Y_yield
    
    # Число фотонов, попавших в фотомножители
    n_hits = pm.Poisson('n_hits', mu=N0_photons * p_geom)
    
    # Полный отклик детектора
    Q = pm.Poisson('Q', mu=n_hits * a_eff)

#С затуханием
with pm.Model() as model_zat:
    N0_photons = E_event * Y_yield

    p_total = p_geom * p_zat
    n_hits_zat = pm.Poisson('n_hits_zat', mu=N0_photons * p_total)
    
    Q_zat = pm.Poisson('Q_zat', mu=n_hits_zat * a_eff)

n_draws = 5000

with model0:
    trace_ideal = pm.sample(draws=n_draws, tune=1000, step=pm.Metropolis(), chains=2, random_seed=42)

with model_zat:
    trace_zat = pm.sample(draws=n_draws, tune=1000, step=pm.Metropolis(), chains=2, random_seed=42)

Q_mcmc_ideal = trace_ideal.posterior['Q'].values.flatten()
Q_mcmc_zat   = trace_zat.posterior['Q_zat'].values.flatten()
n_events = len(Q_mcmc_ideal)

#4 как в задаче 4 и 6 как в задаче 6, те с и без затухания
hits_dir = rnd.binomial(n=rnd.poisson(Y_yield * E_event, size=n_events), p=p_geom)
Q_direct_4 = rnd.poisson(hits_dir * a_eff)

hits_dir_zat = rnd.binomial(n=rnd.poisson(Y_yield * E_event, size=n_events), p=p_geom * p_zat)
Q_direct_6 = rnd.poisson(hits_dir_zat * a_eff)

# Считаем статистики
mean_mcmc = np.mean(Q_mcmc_ideal)
std_mcmc = np.std(Q_mcmc_ideal)

mean_t4 = np.mean(Q_direct_4)
std_t4 = np.std(Q_direct_4)

mean_t6 = np.mean(Q_direct_6)
std_t6 = np.std(Q_direct_6)

mean_zat = np.mean(Q_mcmc_zat)
std_zat = np.std(Q_mcmc_zat)

print(f"прямой MC: <Q> = {mean_t4:.2f}, sigma = {std_t4:.2f},  sigma/<Q> = {std_t4/mean_t4:.2f}")
print(f"PyMC MCMC: <Q> = {mean_mcmc:.2f} , sigma = {std_mcmc:.2f},  sigma/<Q> = {std_mcmc/mean_mcmc:.2f}")
print(f"С затуханием прямой MC: <Q> = {mean_t6:.2f}, sigma = {std_t6:.2f},  sigma/<Q> = {std_t6/mean_t6:.2f}")
print(f"С затуханием PyMC: <Q> = {mean_zat:.2f}, sigma = {std_zat:.2f}, sigma/<Q> = {std_zat/mean_zat:.2f}")



# Сравнение PyMC MCMC и прямой MC без затухания
plt.hist(Q_direct_4, bins=45, alpha=0.5, label="Прямой MC без з.", edgecolor='black')
plt.hist(Q_mcmc_ideal, bins=45, alpha=0.5, label="PyMC MCMC без з.", edgecolor='black')
plt.axvline(125, color="red", linestyle="--", label="Теория: 125")
plt.xlabel("Заряд Q")
plt.ylabel("Частота")
plt.title("Без затухания")
plt.legend()
plt.tight_layout()
plt.show()

# Сравнение PyMC MCMC и прямой MC С затуханиЕМ
plt.hist(Q_direct_6, bins=45, alpha=0.5, label="Прямой MC с з.", edgecolor='black')
plt.hist(Q_mcmc_zat, bins=45, alpha=0.5, label="PyMC MCMC c з.", edgecolor='black')
plt.axvline(125 * p_zat, color="green", linestyle="--", label=f"Теория {125 * p_zat:.1f}")
plt.xlabel("Заряд Q")
plt.ylabel("Частота")
plt.title("С затуханием")
plt.legend()

plt.tight_layout()
plt.show()
