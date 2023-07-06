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
Run this script to reproduce all the figures contained in:

G. Carlier, E. Chenchene, K. Eichinger.
Wasserstein medians: robustness, PDE characterization and numerics,
2023. DOI: 10.48550/arXiv.2307.01765.
"""

import pathlib
import experiments as exps


if __name__ == "__main__":

    pathlib.Path("Figures").mkdir(parents=True, exist_ok=True)

    print('\n\n*** Reproducing Figure 1.\n')

    exps.london_barycenters_and_medians()

    print("\n\n*** Reproducing Figure 2.\n")

    exps.compare_with_sinkhorn()

    print("\n\n*** Reproducing Figure 4.\n")

    # comparing vertical medians
    exps.compare_medians()

    # comparing horizontal medians
    exps.compare_h_medians()

    # medians computed with DRS: consider reducing the dimension or the maximum
    # number of iterations to get the results faster.
    print("\n\n*** Reproducing Figure 6.\n")

    exps.drs_with_flows()

    print("\n\n*** Reproducing Figure 3.\n")

    exps.drs_for_images()
