# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as pl
import pylibsparseir as sparse_ir

# %%
beta = 40.0
wmax = 2.0
basis = sparse_ir.FiniteTempBasis('F', beta, wmax, eps=2e-8)

# %%
pl.semilogy(basis.s / basis.s[0], '-+')
pl.title(r'singular values $s_\ell/s_0$ of $K(\tau, \omega)$')
pl.xlabel(r'index $\ell$');


# %%
def semicirc_dos(w):
    return 2/np.pi * np.sqrt((np.abs(w) < wmax) * (1 - np.square(w/wmax)))

def insulator_dos(w):
    return semicirc_dos(8*w/wmax - 4) + semicirc_dos(8*w/wmax + 4)

# For testing, compute exact coefficients g_l for two models
rho1_l = basis.v.overlap(semicirc_dos)
rho2_l = basis.v.overlap(insulator_dos)
g1_l = -basis.s * rho1_l
g2_l = -basis.s * rho2_l

# Put some numerical noise on both of them (30% of basis accuracy)
rng = np.random.RandomState(4711)
noise = 0.3 * basis.s[-1] / basis.s[0]
g1_l_noisy = g1_l + rng.normal(0, noise, basis.size) * np.linalg.norm(g1_l)
g2_l_noisy = g2_l + rng.normal(0, noise, basis.size) * np.linalg.norm(g2_l)

# %%
# Analytic continuation made (perhaps too) easy
rho1_l_noisy = g1_l_noisy / -basis.s
rho2_l_noisy = g2_l_noisy / -basis.s

# %%
w_plot = np.linspace(-wmax, wmax, 1001)
Vmat = basis.v(w_plot).T
Lprime1 = basis.size // 2

def _plot_one(subplot, dos, rho_l, name):
    pl.subplot(subplot)
    pl.plot(w_plot, dos(w_plot), ":k", label="true")
    pl.plot(w_plot, Vmat[:, :Lprime1] @ rho_l[:Lprime1],
            label=f"reconstructed ($L' = {Lprime1}$)")
    pl.plot(w_plot, Vmat @ rho_l, lw=1,
            label=f"reconstructed ($L' = L = {basis.size}$)")
    pl.xlabel(r"$\omega$")
    pl.title(name)
    pl.xlim(-1.02 * wmax, 1.02 * wmax)
    pl.ylim(-.1, 1)

_plot_one(121, semicirc_dos, rho1_l_noisy, r"semi-elliptic DOS $\rho(\omega)$")
_plot_one(122, insulator_dos, rho2_l_noisy, r"insulating DOS $\rho(\omega)$")
pl.legend()
pl.gca().set_yticklabels([])
pl.tight_layout(pad=.1, w_pad=.1, h_pad=.1)

# %%
# Analytic continuation made (perhaps too) easy
alpha = 100 * noise
invsl_reg = -basis.s / (np.square(basis.s) + np.square(alpha))
rho1_l_reg = invsl_reg * g1_l_noisy
rho2_l_reg = invsl_reg * g2_l_noisy

# %%
# Analytic continuation made (perhaps too) easy
alpha = 100 * noise
invsl_reg = -basis.s / (np.square(basis.s) + np.square(alpha))
rho1_l_reg = invsl_reg * g1_l_noisy
rho2_l_reg = invsl_reg * g2_l_noisy


# %%
def _plot_one(subplot, dos, rho_l, rho_l_reg, name):
    pl.subplot(subplot)
    pl.plot(w_plot, dos(w_plot), ":k", label="true")
    pl.plot(w_plot, Vmat @ rho_l, lw=1, label=f"t-SVD with $L'=L$")
    pl.plot(w_plot, Vmat @ rho_l_reg, label=f"Ridge regression")
    pl.xlabel(r"$\omega$")
    pl.title(name)
    pl.xlim(-1.02 * wmax, 1.02 * wmax)
    pl.ylim(-.1, 1)


_plot_one(121, semicirc_dos, rho1_l_noisy, rho1_l_reg, r"semi-elliptic DOS $\rho(\omega)$")
_plot_one(122, insulator_dos, rho2_l_noisy, rho2_l_reg, r"insulating DOS $\rho(\omega)$")
pl.legend()
pl.gca().set_yticklabels([])
pl.tight_layout(pad=.1, w_pad=.1, h_pad=.1)

# %%
dos3 = np.array([-0.6, -0.1, 0.1, 0.6]) * wmax
rho3_l = basis.v(dos3).sum(1)
g3_l = -basis.s * rho3_l


# %%
def _plot_one(subplot, g_l, rho_l, title):
    pl.subplot(subplot)
    n = np.arange(0, g_l.size, 2)
    pl.semilogy(n, np.abs(g_l[::2]/g_l[0]), ':+b', label=r'$|G_\ell/G_0|$')
    pl.semilogy(n, np.abs(rho_l[::2]/rho_l[0]), ':xr', label=r'$|\rho_\ell/\rho_0|$')
    pl.title(title)
    pl.xlabel('$\ell$')
    pl.ylim(1e-5, 2)

_plot_one(121, g1_l, rho1_l, r'semielliptic $\rho(\omega)$')
pl.legend()
_plot_one(122, g3_l, rho3_l, r'discrete $\rho(\omega)$')
pl.gca().set_yticklabels([])
pl.tight_layout(pad=.1, w_pad=.1, h_pad=.1)

# %%
import scipy.stats as sp_stats

# %%
f = sp_stats.cauchy(scale=0.1 * np.pi / beta).pdf
pl.plot(w_plot, f(w_plot))
pl.title("Cauchy distribution");

# %%
w = np.linspace(-wmax, wmax, 21)
K = -basis.s[:, None] * np.array(
        [basis.v.overlap(lambda w: f(w - wi)) for wi in w]).T

# %%
