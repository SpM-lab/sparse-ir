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
import sys
import math

# %%
T=0.1
wmax =1

beta = 1/T
eps = 1e-10

# %%
#construction of the Kernel K

#Fermionic Basis

basisf = sparse_ir.FiniteTempBasis('F', beta, wmax, eps)

matsf=sparse_ir.MatsubaraSampling(basisf)
tausf=sparse_ir.TauSampling(basisf)

#Bosonic Basis

basisb = sparse_ir.FiniteTempBasis('B', beta, wmax, eps)

matsb=sparse_ir.MatsubaraSampling(basisb)
tausb=sparse_ir.TauSampling(basisb)


# %%
def rho(x):
    return 2/np.pi*np.sqrt(1-(x/wmax)**2)

rho_l=basisf.v.overlap(rho)

G_l_0=-basisf.s*rho_l


#We compute G_iw two times as we will need G_iw_0 as a constant later on

G_iw_0=matsf.evaluate(G_l_0)
G_iw_f=matsf.evaluate(G_l_0)

# %%

# %%

# %%
