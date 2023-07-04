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
This file contains useful functions to plot our numerical results.
For details and references, see Section 7 in:

G. Carlier, E. Chenchene, K. Eichinger.
Wasserstein medians: robustness, PDE characterization and numerics,
2023. DOI: xx.xxxxx/arXiv.xxxx.yyyyy.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import data as dt


def plot_sample(NUs):

    n, N = np.shape(NUs)
    p = int(np.sqrt(n))

    cmaps = ["Blues", "Reds", "Greens", "Oranges", "Purples"]

    plt.figure(figsize=(35, 35))

    for k in range(N):
        nu_plot = np.copy(NUs[:, k])
        nu_plot[nu_plot < 1e-5] = np.nan

        plt.imshow(np.reshape(nu_plot, (p, p)), cmap=cmaps[k], alpha=0.5)

    plt.xticks([])
    plt.yticks([])
    plt.savefig("Figures/sample_intro.pdf", bbox_inches='tight')
    plt.show()


def plot_flows(nu, Sig, NUs, p, N):

    n, N = np.shape(NUs)

    nu_plot = np.copy(nu)
    Sig_plot = np.copy(Sig)
    NUs_plot = np.copy(NUs)

    # creating own colormap
    c_white = mcolors.colorConverter.to_rgba('white', alpha=0)
    c_black = mcolors.colorConverter.to_rgba('black', alpha=1)
    c_blue = mcolors.colorConverter.to_rgba('blue', alpha=1)

    rb_lst = [c_white, c_black]
    md_lst = [c_white, c_blue]
    cmap_rb = mcolors.LinearSegmentedColormap.from_list('rb_cmap', rb_lst, 512)
    cmap_md = mcolors.LinearSegmentedColormap.from_list('md_cmap', md_lst, 512)

    fig = plt.figure(figsize=(35, 35))
    ax = fig.add_subplot(1, 1, 1)

    Grid = []
    for i in range(p):
        for j in range(p):
            Grid.append((i, j))

    # zoom selection
    Zoom = dt.read_image("Data/zoom_section.png", p).T
    Zoom = np.reshape(Zoom, p ** 2)
    Zoom = Zoom > 0.5
    Grid_Zoom = np.array(Grid)[Zoom]

    # Rectangle
    min0 = min(Grid_Zoom[:, 0])
    max0 = max(Grid_Zoom[:, 0])
    min1 = min(Grid_Zoom[:, 1])
    max1 = max(Grid_Zoom[:, 1])

    # lenghts
    lx = max0 - min0 + 1
    ly = max1 - min1 + 1

    # sample measures and flows
    for k in range(N):

        sig_plot_k = np.copy(Sig_plot[:, k, :])
        density = np.linalg.norm(sig_plot_k, axis=1)

        plt.imshow(np.reshape(NUs_plot[:, k], (p, p)).T, cmap=cmap_rb, alpha=1)
        plt.imshow(np.reshape(density, (p, p)).T, cmap=cmap_rb, alpha=0.5)

    # wasserstein median
    plt.imshow(np.reshape(nu_plot, (p, p)).T, cmap=cmap_md, vmin=0,
               vmax=np.max(nu), alpha=1)

    # rectangle
    ax.add_patch(Rectangle((min0, min1), lx, ly, linewidth=3, edgecolor='r',
                           facecolor='none'))

    plt.xticks([])
    plt.yticks([])
    plt.savefig('Figures/flows.pdf', bbox_inches="tight")
    plt.show()


def plot_measure(nu, p, title, transpose=True, vmax_in=None):

    plt.figure(figsize=(35, 35))

    if transpose:
        plt.imshow(np.reshape(nu, (p, p)).T, vmin=0, vmax=vmax_in,
                   cmap='Greys')
    else:
        plt.imshow(np.reshape(nu, (p, p)), vmin=0, vmax=vmax_in,
                   cmap='Greys')

    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'Figures/{title}.pdf', bbox_inches="tight")
    plt.show()


def plot_zoom(Sig, NUs, p, N):

    n, N = np.shape(NUs)
    Sig_zoom = np.copy(Sig)
    NUs_zoom = np.copy(NUs)

    # creating own colormap
    c_white = mcolors.colorConverter.to_rgba('white', alpha=0)
    c_black = mcolors.colorConverter.to_rgba('black', alpha=1)
    sm_lst = [c_white, c_black]
    cmap_sm = mcolors.LinearSegmentedColormap.from_list('sm_cmap', sm_lst, 512)

    Grid = []
    for i in range(p):
        for j in range(p):
            Grid.append((i, j))

    # reading zoom area
    Zoom = dt.read_image("Data/zoom_section.png", p).T
    Zoom = np.reshape(Zoom, p ** 2)
    Zoom = Zoom > 0.5
    Grid_Zoom = np.array(Grid)[Zoom]

    # building rectangle
    min0 = min(Grid_Zoom[:, 0])
    max0 = max(Grid_Zoom[:, 0])
    min1 = min(Grid_Zoom[:, 1])
    max1 = max(Grid_Zoom[:, 1])

    # lengths
    lx = max0 - min0 + 1
    ly = max1 - min1 + 1

    Grid_Zoom_sig = []
    for i in range(lx):
        for j in range(ly):
            Grid_Zoom_sig.append((i, j))
    Grid_Zoom_sig = np.array(Grid_Zoom_sig)

    Grid_Zoom_sig_Active = []
    for i in range(lx):
        for j in range(ly):
            if i % 6 == 0 and j % 6 == 0:
                Grid_Zoom_sig_Active.append(True)
            else:
                Grid_Zoom_sig_Active.append(False)

    plt.figure(figsize=(35, 35))

    for k in range(N):

        nu_plot_k = np.copy(NUs_zoom[:, k][Zoom])
        sig_plot_k = np.copy(Sig_zoom[:, k, :][Zoom, :])
        density = np.linalg.norm(sig_plot_k, axis=1)
        plt.imshow(np.reshape(nu_plot_k, (lx, ly)).T, cmap=cmap_sm, alpha=1)
        plt.imshow(np.reshape(density, (lx, ly)).T, cmap=cmap_sm, alpha=0.5)
        plt.quiver([p[0] for p in Grid_Zoom_sig[Grid_Zoom_sig_Active]],
                   [p[1] for p in Grid_Zoom_sig[Grid_Zoom_sig_Active]],
                   - sig_plot_k[Grid_Zoom_sig_Active, 1],
                   sig_plot_k[Grid_Zoom_sig_Active, 0],
                   color='black', width=0.003, alpha=1, scale=0.2)

        plt.yticks([])
        plt.xticks([])

    plt.savefig('Figures/zoom_section.pdf', bbox_inches='tight')
    plt.show()


def plot_sample_nonover(NUs, experiment_number, vmax_in=None):

    n, N = np.shape(NUs)
    p = int(np.sqrt(n))

    fig, axs = plt.subplots(1, 3, figsize=(35, 35))

    for k in range(N):

        axs[k].imshow(np.reshape(NUs[:, k], (p, p)).T,
                      vmin=0, vmax=vmax_in, cmap='Greys')
        axs[k].set_yticklabels([])
        axs[k].set_xticklabels([])

    fig.tight_layout()
    plt.savefig(f"Figures/sample_single_{experiment_number}.pdf",
                bbox_inches="tight")
    plt.show()
