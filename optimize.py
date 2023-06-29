# -*- coding: utf-8 -*-
#
#    Copyright (C) 2023 Guillaume Carlier (carlier@ceremade.dauphine.fr)
#                       Enis Chenchene (enis.chenchene@uni-graz.at)
#                       Katharina Eichinger (eichinger@ceremade.dauphine.fr)
#
#    This file is part of the example code repository for the paper:
#
#      G. Carlier, E. Chenchene, K. Eichinger.
#      Wasserstein medians: robustness, PDE characterization and numerics,
#      2023. DOI: xx.xxxxx/arXiv.xxxx.yyyyy
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains the functions we used to approximate Wasserstein medians.
For details and references, see section 7 in:

G. Carlier, E. Chenchene, K. Eichinger.
Wasserstein medians: robustness, PDE characterization and numerics,
2023. DOI: xx.xxxxx/arXiv.xxxx.yyyyy.
"""

import numpy as np
import structures as st
import plot as show
import scipy.sparse as sp
from tqdm import trange


def compute_1d_barycenter(Fs, X):

    n, N = np.shape(Fs)

    Qs = []
    Ts = []

    for k in range(N):

        T, Q = st.pseudo_invert(X, Fs[:, k])
        Qs.append(Q)
        Ts.append(T)

    X_bar, F_bar = st.mean_of_quantiles(Ts, Qs, X)

    return X_bar, st.make_measure_cdf(F_bar), F_bar


def bar_sinkhorn(NUs, C, eps, maxit=10):

    n, N = np.shape(NUs)
    K = np.exp(- C / eps)

    u = np.ones((n, N))

    loop_obj = trange(maxit)

    for iters in loop_obj:

        v = NUs / (K.T @ u)
        bar = np.prod((K @ v) ** (1 / N), axis=1)
        u = bar[:, np.newaxis] / (K @ v)

    return bar


def douglas_rachford_medians(NUs, maxit, tau, experiment=1, number=1):

    n, N = np.shape(NUs)
    p = int(np.sqrt(n))
    ones = np.ones(N)

    # create divergence and gradient operators
    Dx = st.create_sparse_gradx_mat(p)
    Dy = st.create_sparse_grady_mat(p)
    M1 = -Dx.T
    M2 = -Dy.T

    # Laplacian
    Lap = -(Dx.T @ Dx + Dy.T @ Dy)
    Lap_str = sp.eye(n) - 1 / N * Lap

    # initialize variables
    W_sig = np.zeros((n, N, 2))
    W_nu = np.ones(n) / n

    loop_obj = trange(maxit)
    res = 0

    # iterations
    for k in loop_obj:

        Sig_0 = st.prox_l1(tau, W_sig, n, N)
        nu_0 = st.projection_simplex_sort(W_nu)

        xi = st.proj_div(2 * Sig_0 - W_sig, 2 * nu_0 - W_nu, Lap, Lap_str, NUs,
                         M1, M2, Dx, Dy, ones, N, n)
        W_sig_old = np.copy(W_sig)
        W_sig = Sig_0 + st.grad_mat(xi, Dx, Dy, n, N)
        W_nu_old = np.copy(W_nu)
        W_nu = nu_0 + np.sum(xi, axis=1)

        if k % 10 == 0:

            res = np.sum(np.square(W_sig_old - W_sig)) + \
                np.sum(np.square(W_nu_old - W_nu))

        if k % 500 == 0 and k >= 30:

            if experiment == 1:
                show.plot_flows(nu_0, Sig_0, NUs, p, N)
                show.plot_zoom(Sig_0, NUs, p, N)
            else:
                show.plot_measure(nu_0, p, f'median_single_{number}',
                                  vmax_in=1e-4 * 0.4)

        loop_obj.set_postfix_str(f'Residual at iter. {k - k % 10}={res}')

    if experiment == 1:
        show.plot_flows(nu_0, Sig_0, NUs, p, N)
        show.plot_zoom(Sig_0, NUs, p, N)
    else:
        show.plot_measure(nu_0, p, f'median_single_{number}',
                          vmax_in=1e-4 * 0.4)
