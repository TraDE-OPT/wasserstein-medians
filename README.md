# Wasserstein medians: robustness, PDE characterization and numerics. Example code

This repository contains the experimental source code to reproduce the numerical experiments in:

* G. Carlier, E. Chenchene, K. Eichinger. Wasserstein medians: robustness, PDE characterization and numerics, 2023. [ArXiv preprint](https://arxiv.org/abs/2307.01765)

To reproduce the results of the numerical experiments in Section 5, run:
```bash
python3 main.py
```

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@article{cce2023,
  author = {Carlier, Guillaume and Chenchene, Enis and Eichinger, Katharina},
  title = {Wasserstein medians: robustness, {PDE} characterization and numerics},
  pages = {2307.01765},
  journal = {ArXiv},
  year = {2023}
}
```

## Requirements

Please make sure to have the following Python modules installed, most of which should be standard.

* [numpy>=1.20.1](https://pypi.org/project/numpy/)
* [scipy>=1.6.2](https://pypi.org/project/scipy/)
* [matplotlib>=3.3.4](https://pypi.org/project/matplotlib/)
* [pandas>=1.5.3](https://pandas.pydata.org)
* [tqdm >=4.65.0](https://tqdm.github.io)
* [pathlib>=1.0.1](https://pathlib.readthedocs.org/)

## Highlights
* Exact routine to compute **one dimensional** Wasserstein barycenters, see:
    ```python
     X_bar, bar, F_bar = compute_1d_barycenter(Fs, X)
    ```
    The parameters of the function are:
    
    * `Fs`: Cumulative distribution functions of the sample measures.
    * `X`: Space grid that supports the sample measures.
    
    **Note**: The function can be adapted to compute 1d _horizontal selections_ of Wasserstein medians modifying `mean_of_quantiles` in a suitable way.

* **Douglas-Rachford** Splitting method to compute Wasserstein medians between **images**, see:
    ```python
     douglas_rachford_medians(NUs, maxit, tau)
    ```
    The parameters of the function are:
    
    * `NUs`: Sample images.
    * `maxit`: Maximal number of iterations allowed.
    * `tau`: Douglas-Rachford Splitting step-size.
    
    **Note**: return `nu_0` to get a Wasserstein median.


## Acknowledgments  

* | ![](<euflag.png>) | EC has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no. 861137. KE acknowledges that this project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 754362. |
  |----------|----------|
* | GC acknowledges the support of the Lagrange Mathematics and Computing Research Center. |
  |----------|
* The dataset used to generate Figure 1 has been downloaded from https://tfl.gov.uk/info-for/open-data-users/. Powered by TfL Open Data. Contains OS data © Crown copyright and database rights 2016.
* All other data used for numerical experiments in this project have been created artificially by the authors.

## License  
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
