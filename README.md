# Output-feedback stochastic MPC

This repository documents the code (Python) used for the numerical illustration in the paper

Florian Messerer, Katrin Baumgärtner, Moritz Diehl. A Dual-control effect preserving formulation for nonlinear output-feedback stochastic model predictive control with constraints. https://arxiv.org/abs/2209.07973, 2022.

## Overview

* `src\` : folder containing the code 
    * `src\robot_nl_run_example.py` : main file to run the example
    * `src\robot_nl_niceplots.py`   : generates the corresponding plots
    * `src\robot_nl_results\res_2022-11-22-15-00-08.npy`: results of the example run visualized in the paper
* `requirements.txt` : file containing the required packages. You can install them by running `pip install -r requirements.txt`
* `package_versions.txt`: file documenting the versions of the packages used for the experiments
* `README.md` : this file
* `LICENSE` : license file
