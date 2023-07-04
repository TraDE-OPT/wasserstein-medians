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
Data processing.
For details and references, see section 7 in:

G. Carlier, E. Chenchene, K. Eichinger.
Wasserstein medians: robustness, PDE characterization and numerics,
2023. DOI: xx.xxxxx/arXiv.xxxx.yyyyy.
"""


import numpy as np
from PIL import Image
import pandas as pd


def read_dataset():

    data = pd.read_csv("Data/london_underground.csv", sep=';')
    Times = np.array(data.keys()[4:])
    NUs = np.array(pd.read_csv("Data/london_underground.csv", sep=';'))[:, 4:]
    sums = np.sum(NUs, axis=1)
    NUs[sums != 0] /= sums[sums != 0][:, np.newaxis]

    return (NUs[sums != 0].T).astype("float64"), Times


def read_image(img_name, p):

    img = Image.open(img_name).convert('L')
    Img = 255 - np.array(img.resize((p, p)))

    return Img


def read_images(Data, p):

    N = len(Data)
    NUs = np.zeros((p ** 2, N))

    for i in range(N):

        img = read_image(Data[i], p)
        new_image = np.reshape(img, p ** 2)
        mass = np.sum(new_image)
        NUs[:, i] = new_image / mass

    return NUs


def create_domain(p):

    Xs = np.zeros((p, p, 2))

    for i in range(p):
        for j in range(p):

            Xs[i, j, 0] = i
            Xs[i, j, 1] = j

    Xs = np.reshape(Xs, (p ** 2, 2))
    Xs = Xs / p

    return Xs


def gaussian(x, mu, sig):

    return np.exp(- (x - mu) ** 2 / (2 * sig ** 2))


def create_gaussian(X, mu, sig):

    M = gaussian(X, mu, sig)

    return M / np.sum(M)
