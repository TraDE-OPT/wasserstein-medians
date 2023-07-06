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
#      2023. DOI: 10.48550/arXiv.2307.01765
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
This file contains the functions we used to generate the images in:

G. Carlier, E. Chenchene, K. Eichinger.
Wasserstein medians: robustness, PDE characterization and numerics,
2023. DOI: 10.48550/arXiv.2307.01765.
"""

import numpy as np
import structures as st
import plot as show
import matplotlib.pyplot as plt
import data as dt
from tqdm import trange
import optimize as op


def london_barycenters_and_medians():

    _fig, axs = plt.subplots(2, 5, figsize=(24, 6))

    nums = [9, 29, 39, 59, 81]
    loop_obj = trange(5)

    for i in loop_obj:

        N = nums[i]
        loop_obj.set_description(f"Num. Stations: {N}")

        NUs, Times = dt.read_dataset()
        Times = [t[:2] + ":" + t[2:4] for t in Times]
        NUs = NUs[:, :N]
        n, N = np.shape(NUs)
        X_meas = np.array(range(n))
        Fs = st.from_sample_to_cdf(NUs)

        loop_obj.set_postfix_str('Computing median...')

        # median
        F_med = st.make_median_cdf(Fs, "half")
        med = st.make_measure_cdf(F_med)

        loop_obj.set_postfix_str('Computing barycenter...')

        # barycenter
        X_bar, bar, F_bar = op.compute_1d_barycenter(Fs, X_meas)

        loop_obj.set_postfix_str('Plotting results...')

        # plotting densities and cdfs
        for k in range(N):
            axs[0, i].plot(X_meas, NUs[:, k], alpha=0.1)
            axs[0, i].set_xticks([])
            axs[0, i].set_ylim(0, 0.1)
            if i != 0:
                axs[0, i].set_yticks([])
            axs[1, i].plot(Fs[:, k], alpha=0.1)
            axs[1, i].set_xticks(X_meas[::15], Times[::15])

            if i != 0:
                axs[1, i].set_yticks([])

        axs[0, i].plot(X_meas, med, color="blue")
        axs[0, i].plot(X_bar, st.make_measure_cdf(F_bar), color="black")
        axs[1, i].plot(X_meas, F_med, color="blue")
        axs[1, i].plot(X_bar, F_bar, color="k")

    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.savefig("Figures/london_underground.pdf", bbox_inches="tight")
    plt.show()

    print("\n* Results saved in Figures as\n---> london_underground.pdf")


def compare_with_sinkhorn():

    p = 120  # image sizes
    eps = 1e-3

    Data_intro = ["Data/intro1.png", "Data/intro2.png", "Data/intro3.png",
                  "Data/intro4.png", "Data/intro5.png"]

    NUs = dt.read_images(Data_intro, p)
    show.plot_sample(NUs)

    Xs = dt.create_domain(p)

    print("Computing Wasserstein barycenter with Sinkhorn:\n")
    C = st.make_C(Xs, 2)
    bar = op.bar_sinkhorn(NUs, C, eps, maxit=100)
    show.plot_measure(bar, p, title="barycenter_intro", transpose=False)

    print("\nComputing Wasserstein median with Sinkhorn:\n")
    C = st.make_C(Xs, 1)
    med = op.bar_sinkhorn(NUs, C, 2 * 1e-3, maxit=100)
    show.plot_measure(med, p, title="median_intro", transpose=False)

    print("\n* Results saved in Figures as\n" +
          "---> sample_intro.pdf, barycenter_intro.pdf and median_intro.pdf.")


def compare_medians():

    mesh_size = 200

    # creating sample
    X_meas = np.linspace(-10, 10, mesh_size)
    NUs = np.zeros((mesh_size, 4))
    NUs[:, 0] = dt.create_gaussian(X_meas, 2, 2.5)
    NUs[:, 1] = dt.create_gaussian(X_meas, 2, 1.5)
    NUs[:, 2] = dt.create_gaussian(X_meas, -2, 2.5)
    NUs[:, 3] = dt.create_gaussian(X_meas, -2, 1.5)

    # turn sample into cdfs
    Fs = st.from_sample_to_cdf(NUs)

    _fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    ticks_label_size = 15

    # lower median
    F = st.make_median_cdf(Fs, "lower")
    nu = st.make_measure_cdf(F)

    # plotting lower median
    for i in range(4):
        mu = np.copy(NUs[:, i])
        axs[0, 0].fill_between(X_meas, mu, alpha=0.3)

    axs[0, 0].fill_between(X_meas, nu, color='Black', alpha=0.3)
    axs[0, 0].set_xticks([])
    axs[0, 0].tick_params(axis='both', labelsize=ticks_label_size)
    axs[0, 0].set_ylim([0, 0.03])

    axs[1, 0].plot(X_meas, F, color="Black")
    for i in range(4):
        axs[1, 0].plot(X_meas, Fs[:, i], alpha=0.7)
    axs[1, 0].tick_params(axis='both', labelsize=ticks_label_size)

    # half median
    F = st.make_median_cdf(Fs, "half")
    nu = st.make_measure_cdf(F)

    # plotting lower median
    for i in range(4):
        mu = np.copy(NUs[:, i])
        axs[0, 1].fill_between(X_meas, mu, alpha=0.3)

    axs[0, 1].fill_between(X_meas, nu, color='Black', alpha=0.3)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_ylim([0, 0.03])

    axs[1, 1].plot(X_meas, F, color="Black")
    for i in range(4):
        axs[1, 1].plot(X_meas, Fs[:, i], alpha=0.7)
    axs[1, 1].set_yticks([])
    axs[1, 1].tick_params(axis='both', labelsize=ticks_label_size)

    # upper median
    F = st.make_median_cdf(Fs, "upper")
    nu = st.make_measure_cdf(F)

    # plotting lower median
    for i in range(4):
        mu = np.copy(NUs[:, i])
        axs[0, 2].fill_between(X_meas, mu, alpha=0.3)

    axs[0, 2].fill_between(X_meas, nu, color='Black', alpha=0.3)
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_ylim([0, 0.03])

    axs[1, 2].plot(X_meas, F, color="Black")
    for i in range(4):
        axs[1, 2].plot(X_meas, Fs[:, i], alpha=0.7)
    axs[1, 2].set_yticks([])
    axs[1, 2].tick_params(axis='both', labelsize=ticks_label_size)

    plt.savefig("Figures/v_medians_one_d.pdf", bbox_inches='tight')
    plt.show()


def compare_h_medians():

    mesh_size = 200

    # creating sample
    X_meas = np.linspace(-10, 10, mesh_size)
    NUs = np.zeros((mesh_size, 4))
    NUs[:, 0] = dt.create_gaussian(X_meas, 5, 1.7)
    NUs[:, 1] = dt.create_gaussian(X_meas, 2, 1.7)
    NUs[:, 2] = dt.create_gaussian(X_meas, -5, 1.7)
    NUs[:, 3] = dt.create_gaussian(X_meas, -2, 1.7)

    # horizontal half median
    med_h = np.array(dt.create_gaussian(X_meas, 0, 1.7))
    F_h = np.cumsum(med_h)

    # turn sample into cdfs
    Fs = st.from_sample_to_cdf(NUs)

    # vertical 1/2 median
    F_v = st.make_median_cdf(Fs, "half")
    med_v = st.make_measure_cdf(F_v)

    _fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ticks_label_size = 15

    # plotting horizontal median
    for i in range(4):
        mu = np.copy(NUs[:, i])
        axs[0, 0].fill_between(X_meas, mu, alpha=0.3)

    axs[0, 0].fill_between(X_meas, med_h, color='Black', alpha=0.3)
    axs[0, 0].set_xticks([])
    axs[0, 0].tick_params(axis='both', labelsize=ticks_label_size)
    axs[0, 0].set_ylim([0, 0.025])

    axs[1, 0].plot(X_meas, F_h, color="Black")
    for i in range(4):
        axs[1, 0].plot(X_meas, Fs[:, i], alpha=0.7)
    axs[1, 0].tick_params(axis='both', labelsize=ticks_label_size)

    # plotting vertical median
    for i in range(4):
        mu = np.copy(NUs[:, i])
        axs[0, 1].fill_between(X_meas, mu, alpha=0.3)

    axs[0, 1].fill_between(X_meas, med_v, color='Black', alpha=0.3)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_ylim([0, 0.025])

    axs[1, 1].plot(X_meas, F_v, color="Black")
    for i in range(4):
        axs[1, 1].plot(X_meas, Fs[:, i], alpha=0.7)
    axs[1, 1].set_yticks([])
    axs[1, 1].tick_params(axis='both', labelsize=ticks_label_size)

    plt.savefig("Figures/v_vs_h_medians_one_d.pdf", bbox_inches='tight')
    plt.show()

    print("* Results saved in Figures as\n" +
          "---> v_medians_one_d.pdf, v_vs_h_medians_one_d.pdf")


def drs_with_flows():

    p = 420  # image sizes
    maxit = 10000  # maximum number of iterations

    print("Note: current iterate displayed every 500 iterations.\n" +
          "      This phase might take several hours.\n")

    Data_exp1 = ["Data/img21.png", "Data/img22.png", "Data/img23.png"]

    # reading data
    NUs = dt.read_images(Data_exp1, p)

    # computing wasserstein median
    op.douglas_rachford_medians(NUs, maxit, tau=1e-1, experiment=1)

    print("\n* Result saved in Figures as\n" +
          "---> flows.pdf and zoom_section.pdf")


def drs_for_images():

    p = 420  # image sizes
    maxit = 2000  # maximum number of iterations

    print("Note: Reproducing median in row 1.\n" +
          "      This phase might take several hours.\n")

    Data_exp2 = ["Data/sp1_maxi.png", "Data/sp2_maxi.png", "Data/sp3_maxi.png"]

    # reading data
    NUs = dt.read_images(Data_exp2, p)

    # showing sample
    show.plot_sample_nonover(NUs, 1, vmax_in=1e-4 * 0.4)

    # computing wasserstein median
    op.douglas_rachford_medians(NUs, maxit, tau=1e-2, experiment=2, number=1)

    print("\n* Result of row 1 saved in Figures as\n" +
          "---> sample_single_1.pdf and median_single_1.pdf")

    print("\nNote: Reproducing median in row 2.\n" +
          "      This phase might take several hours.\n")

    Data_exp2 = ["Data/sh1.png", "Data/sh2.png", "Data/sh3.png"]

    # reading data
    NUs = dt.read_images(Data_exp2, p)

    # showing sample
    show.plot_sample_nonover(NUs, 2, vmax_in=1e-4 * 0.4)

    # computing wasserstein median
    op.douglas_rachford_medians(NUs, maxit, tau=1e-2, experiment=2, number=2)

    print("\n* Result of row 2 saved in Figures as\n" +
          "---> sample_single_2.pdf and median_single_2.pdf")

    print("\nNote: Reproducing median in row 3.\n" +
          "      This phase might take several hours.\n")

    Data_exp3 = ["Data/sh4.png", "Data/sh5.png", "Data/sh6.png"]

    # reading data
    NUs = dt.read_images(Data_exp3, p)

    # showing sample
    show.plot_sample_nonover(NUs, 3, vmax_in=1e-4*0.4)

    # computing wasserstein median
    op.douglas_rachford_medians(NUs, maxit, tau=1e-2, experiment=2, number=3)

    print("\n* Result of row 3 saved in Figures as\n" +
          "---> sample_single_3.pdf and median_single_3.pdf")
