import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt


def MHM(f_sampler, pi_pdf, x0):
    x = x0
    rejection_count = 0
    x_list = [x0]
    for _ in range(1000):
        eps = f_sampler()
        if np.random.rand()>0.5:
            y = x/eps
        else:
            y = x*eps
        alpha = min(1, pi_pdf(y)/pi_pdf(x)*abs(y/x))
        if np.random.rand() < alpha:
            x = y
        else:
            rejection_count += 1
        x_list.append(x)
    return x_list


def qq_plot(x_list, pi_ppf):
    n = len(x_list)
    ordered = np.sort(x_list)
    plt.plot(ordered, [pi_ppf(i/n) for i in range(n)], '.')
    ax = plt.gca()
    plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    plt.grid()
    plt.xlabel('Approximated quantiles')
    plt.ylabel('True quantiles')
    plt.show()


def histo_plot(x_list, pi_pdf):
    plt.hist(x_list, bins=30, density=True)
    rng = np.linspace(min(x_list), max(x_list), 100)
    plt.plot(rng, [pi_pdf(x) for x in rng], 'r')
    plt.show()



uni_f = lambda: np.random.rand()*2-1
gauss_f = lambda: min(1, max(-1, np.random.randn()))
pi_1 = stat.norm.pdf
pi_2 = stat.expon.pdf

x_list = MHM(uni_f, pi_2, 1.)
histo_plot(x_list, stat.expon.pdf)