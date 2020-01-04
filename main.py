import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from tqdm import tqdm


# Generic SA
def SA(gamma, H, K, x_0, theta_0, P, T_max=20):
    zeta = 0
    i = 0
    n = 0
    x = x_0
    theta = theta_0
    x_list = [x_0]
    theta_list = [theta_0]
    for n in tqdm(range(T_max)):
        if i + zeta > len(gamma)-1 or i > len(K) -1:
            return x_list, theta_list, n, i, zeta, False
        x = P(theta, x)
        theta += gamma[i+zeta]*H(theta, x)
        if isinstance(x, np.ndarray):
            x_list.append(x.copy())
        else:
            x_list.append(x)
        if isinstance(theta, np.ndarray):
            theta_list.append(theta.copy())
        else:
            theta_list.append(theta)
        if theta in K[i]:
            zeta += 1
        else:
            i += 1
            zeta = 0
    return x_list, theta_list, n, i, zeta, True


class Compact:
    def __init__(self, M):
        self.M = M

    def __contains__(self, item):
        if isinstance(item, np.ndarray):
            return np.linalg.norm(item, ord=2) < self.M
        else:
            return abs(item) < self.M


# Calls SA for quantile estimation
def qt_sa(phi, P, q, x_0, theta_0, T_max=10000):

    def H(theta,x):
        if phi(x) <= theta:
            return q-1
        return q

    K = [Compact(10**(5+i)) for i in range(100)]
    gamma = [1./(1+i)**0.8 for i in range(int(1e4))]

    return SA(gamma, H, K, x_0, theta_0, P, T_max)


# Symetric Metropolis Hasting transition
def SMH_P(T_spler, pi_pdf):
    def P(theta, x):
        z = T_spler(x)
        alpha = min(1, pi_pdf(z)/pi_pdf(x))
        if np.random.rand() < alpha:
            return z
        return x
    return P


# Dimension 1 example of quantile with SA
def d1_qt_sa():
    phi = lambda x: x

    mean = np.array([5])
    cov = np.array([1])
    pi_pdf = stat.norm(loc=mean, scale=cov).pdf
    T_spler = lambda x: stat.multivariate_normal(mean=x, cov=0.1).rvs()
    P = SMH_P(T_spler, pi_pdf)

    q = 0.8
    x_0 = 1.
    theta_0 = 1.
    x_list, theta_list, n, i, zeta, term = qt_sa(phi, P, q, x_0, theta_0, 10000)
    print('Target: ', stat.norm(loc=mean, scale=cov).ppf(q))
    print('Mean: ', np.mean(theta_list[9000:]))
    f, axs = plt.subplots(1,2)
    axs[0].hist(x_list, bins=30, density=True)
    rng = np.linspace(min(x_list), max(x_list), 100)
    axs[0].plot(rng, [pi_pdf(x) for x in rng], 'r')
    axs[1].plot(range(len(theta_list)), theta_list)
    plt.show()


# Dimension 2 example of quantile with SA
def d2_qt_sa():
    M = np.array([[1, 2]])
    b = -5
    phi = lambda x: M@x+b

    mean = np.array([-2,3])
    cov = np.array([[3,1],[1,3]])
    pi_pdf = stat.multivariate_normal(mean=mean, cov=cov).pdf
    T_spler = lambda x: stat.multivariate_normal(mean=x).rvs()
    P = SMH_P(T_spler, pi_pdf)

    q = 0.8
    x_0 = np.array([1., 1.])
    theta_0 = np.array([1. ,1.])
    qt_sa(phi, P, q, x_0, theta_0)

d1_qt_sa()