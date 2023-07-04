# -*- coding: utf-8 -*-
#
#    Copyright (C) 2023 Guillaume Carlier (carlier@ceremade.dauphine.fr)
#                       Enis Chenchene (enis.chenchene@uni-graz.at)
#                       Katharina Eichinger (katharina.eichinger@polytechnique.edu)
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
This file contains useful functions to reproduce the figures in:

G. Carlier, E. Chenchene, K. Eichinger.
Wasserstein medians: robustness, PDE characterization and numerics,
2023. DOI: xx.xxxxx/arXiv.xxxx.yyyyy.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def mean_of_quantiles(Ts, Qs, Xs):

    N = len(Qs)

    # merging domains
    dom = set()
    for dom_new in Ts:
        dom = dom.union(set(dom_new))

    Q_bar = []
    F_bar = {}
    dom = sorted(list(dom))
    i_low = 0

    for t in dom:

        q_out_t = 0
        for k in range(N):

            if any(Ts[k] >= t):
                q_out_t += 1 / N * Qs[k][Ts[k] >= t][0]
            else:
                q_out_t += 1 / N * Qs[k][-1]

        i_low_old = i_low
        src = Xs[i_low_old:]
        i_low = np.argmin(np.abs(src - q_out_t))
        q_out_t = src[i_low]
        F_bar[q_out_t] = t
        Q_bar.append(q_out_t)

    return np.array(list(F_bar.keys())), np.array(list(F_bar.values()))


def only_jumps(X, F):

    X_out = [X[F > 0][0]]
    F_out = [F[F > 0][0]]

    pos_F = F[F > 0]

    for i, x in enumerate(X[F > 0]):
        if pos_F[i] == F_out[-1]:
            pass
        else:
            X_out.append(x)
            F_out.append(pos_F[i])

    return np.array(X_out), np.array(F_out)


def pseudo_invert(X, F):

    X_out, F_out = only_jumps(X, F)

    return F_out, X_out


def make_C(Xs, expn):

    if expn == 1:
        return np.linalg.norm(Xs[:, :, np.newaxis] - Xs.T[np.newaxis, :, :],
                              axis=1)

    elif expn == 2:
        return np.sum(np.square(Xs[:, :, np.newaxis] - Xs.T[np.newaxis, :, :]),
                      axis=1)


def from_sample_to_cdf(NUs):

    return np.cumsum(NUs, axis=0)


def make_measure_cdf(F):

    return np.diff(F, prepend=0)


def from_cdf_to_sample(Fs):

    return np.diff(Fs, prepend=0, axis=0)


def make_median_cdf(Fs, kind="half"):

    n, N = np.shape(Fs)

    if N % 2 == 0:
        if kind == "half":
            return np.median(Fs, axis=1)
        if kind == "lower":
            return np.median(np.column_stack((-10 * np.ones(n), Fs)), axis=1)
        if kind == "upper":
            return np.median(np.column_stack((Fs, 10 * np.ones(n))), axis=1)
    else:
        return np.median(Fs, axis=1)


def create_sparse_gradx_mat(p):

    diag = np.ones(p)
    diag[-1] = 0
    diag = np.tile(diag, p)

    Dx = sp.spdiags([-diag, [0] + list(diag[:-1])], [0, 1], p ** 2, p ** 2)

    return Dx


def create_sparse_grady_mat(p):

    diag = np.ones(p ** 2)
    diag[-p:] = 0 * diag[-p:]

    up_diag = np.ones(p ** 2)
    up_diag[:p] = 0 * up_diag[:p]

    Dy = sp.spdiags([-diag, up_diag], [0, p], p ** 2, p ** 2)

    return Dy


def grad_mat(Psi, Dx, Dy, n, N):

    Sig_out = np.zeros((n, N, 2))

    Sig_out[:, :, 0] = Dx @ Psi
    Sig_out[:, :, 1] = Dy @ Psi

    return Sig_out


def div_mat(Sig, M1, M2):

    return M1 @ Sig[:, :, 0] + M2 @ Sig[:, :, 1]


def proj_ball(Sig, tau, n, N):

    Sig_out = np.copy(Sig)
    Norms = np.linalg.norm(Sig_out, axis=2)
    Greater = Norms > tau
    Sig_out[Greater, :] = tau * Sig_out[Greater, :] / \
        Norms[Greater, np.newaxis]

    return Sig_out


def prox_l1(tau, Sig, n, N):

    return Sig - proj_ball(Sig, tau, n, N)


def projection_simplex_sort(vv, z=1):

    v = np.copy(vv)
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)

    return w


def proj_div(Sig, nu, Lap, Lap_str, NUs, M1, M2, Dx, Dy, ones, N, n):

    # stabilizing. A possible alternative is precomputing
    # the psuedoinverse of the Laplacian matrix and
    # storing it, but this breaks sparsity.
    nu = nu - (sum(nu) - 1) / n

    xi_p = spl.spsolve(Lap, nu[:, np.newaxis] - div_mat(Sig, M1, M2) - NUs)
    M = spl.spsolve(Lap_str, np.mean(xi_p, axis=1))
    xi = xi_p - M[:, np.newaxis]

    return xi
