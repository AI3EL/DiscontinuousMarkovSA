import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from tqdm import tqdm


def bridge_nwk_sace(d, a, gamma_it, T_max, q, phi):
    g_spl = lambda v : stat.beta.rvs(v,1,size=d)
    g_pdf = lambda v, x: np.prod(stat.beta.pdf(x,v,1))
    p_pdf = lambda z: (np.all(z <= 1) and np.all(z >= 0)).astype(int)

    def Q(theta, u):
        if theta > phi(a,u):
            raise ValueError('Theta not achievable for u')
        new_u = u.copy()
        for i in range(u.shape[0]):
            u_i = new_u.copy()
            u_i[i] = 0
            phi_star = phi(a,u_i)
            inf = max(0, (theta-phi_star)/a[i])

            new_u[i] = (1-inf)*np.random.rand()+inf
        return new_u

    S = lambda u: np.log(u)
    v_hat = lambda s: -np.ones(s.shape)/s


    y = np.ones(d)/2
    z = np.ones(d)/2
    s = S(y)
    v = v_hat(s)
    theta = 1.
    leap_count = 0

    y_list = [y.copy()]
    z_list = [z.copy()]
    s_list = [s.copy()]
    theta_list = [theta]
    v_list = [v.copy()]
    leap_count_list = [leap_count]
    n=0
    while n < T_max:
        gamma = next(gamma_it)
        y = Q(theta, y)
        z = g_spl(v)

        if phi(a,z) < theta:
            theta += gamma*(q-p_pdf(z)/g_pdf(v, z))
        else:
            theta += gamma*q

        if phi(a, y) < theta:
            leap_count += 1
            theta = min(theta, phi(a, y))

            # y = y_list[-1]
            # theta = theta_list[-1]
            # continue

        s = (1-gamma)*s + gamma*S(y)
        v = v_hat(s)

        y_list.append(y.copy())
        z_list.append(z.copy())
        s_list.append(s.copy())
        v_list.append(v.copy())
        theta_list.append(theta)
        leap_count_list.append(leap_count)
        n += 1

        if n % (T_max//20) == 0:
            print('{}%'.format((n*100)//T_max))

    return theta_list, v_list, leap_count_list


# N1 --- x --- x --- x --- N2
def one_bridge(a,u):
    return sum([x*y for x,y in zip(a,u)])


#     -----
# N1X ----- XN2
#     -----
def make_k_prll_bridge(k,d):
    assert not d%k

    def prll_bridge(a,u):
        bridge_lengths = []
        lengths = [x*y for x,y in zip(a,u)]
        for i in range(k):
            bridge_lengths.append(sum(lengths[i*k:(i+1)*k]))
        return min(bridge_lengths)
    return prll_bridge


# Double triangle bridge defined in the paper, only for d=5
def paper_bridge(a,u):
    assert a.shape[0] == 5
    paths = [[0,3], [1,4], [0,2,4], [1,2,3]]
    return min([sum([a[i]*u[i] for i in p]) for p in paths])


def gamma_yielder(v0, pow):
    n = 0
    while True:
        yield v0 / (1 + n) ** pow
        n += 1


d = 5
a = np.array([1,2,3,1,2])
q = 1-10**-3
T_max = int(1e5)
v0 = 0.1
pow = 0.8
gamma_it = gamma_yielder(v0, pow)
theta_list, v_list, leap_count_list = bridge_nwk_sace(d, a, gamma_it, T_max, q, paper_bridge)


# Plotting
f, axs = plt.subplots(1, 2)
axs[0].plot(range(1000, len(theta_list)), theta_list[1000:], label='theta')

mod_leap_count = np.array(leap_count_list) /max(leap_count_list) * max(theta_list)
axs[0].plot(range(1000, len(theta_list)), mod_leap_count[1000:], label='leaps')
axs[0].legend()
axs[0].set_title('Theta and leap count')
v_array = np.array(v_list)
for i in range(d):
    axs[1].plot(range(1000, len(theta_list)), v_array[1000:, i], label='v{}'.format(i+1))
axs[1].legend()
print(v_array[-1])
axs[1].set_title('v parameters')
plt.title('Bridge with v0={}, pow={}'.format(v0,pow))
plt.show()
