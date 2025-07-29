# ML-FSM


[![License](https://img.shields.io/github/license/jonmarks12/ML-FSM)](https://github.com/jonmarks12/ML-FSM/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/jonmarks12/ML-FSM/test.yml?branch=main&logo=github-actions)](https://github.com/jonmarks12/ML-FSM/actions/)
[![Documentation Status](https://readthedocs.org/projects/ml-fsm/badge/?version=latest)](https://ml-fsm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jonmarks12/ML-FSM/branch/dev/graph/badge.svg)](https://codecov.io/gh/jonmarks12/ML-FSM)




[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonmarks12/ML-FSM/blob/main/examples/FSM_GNN_Colab_Example.ipynb)

This repository provides an implementation of the Freezing String Method (FSM) for double-ended transition state searches with internal coordinates interpolation with ML-based potentials.

## Installation

Clone the repository and install:

```bash
git clone https://github.com/jonmarks12/ML-FSM.git
cd ML-FSM
pip install .
```

## Example

To run the FSM with default parameters and the ASE EMT calculator on a Diels Alder reaction run:
```python
python examples/fsm_example.py data/sharada/06_diels_alder/ --calculator emt
```
Note: Users are responsible for installing their desired quantum chemistry backend, current calculators supported in fsm_example.py are [SchNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.SchNet.html), [AIMNet2](https://github.com/isayevlab/AIMNet2), [MACEOFF23](https://github.com/ACEsuit/mace-off), [FAIR UMA](https://github.com/facebookresearch/fairchem), [TensorNet](https://github.com/torchmd/torchmd-net), [xTB](https://github.com/grimme-lab/xtb), [QChem](https://www.q-chem.com).

To use custom calculators setup your own script, a basic example script/notebook is shown in examples/FSM_GNN_Colab_Example.ipynb

## Usage
For projects referencing algorithmic improvements to the FSM please cite:

Marks, J., & Gomes, J. (2024). Incorporation of Internal Coordinates Interpolation into the Freezing String Method. http://arxiv.org/abs/2407.09763

For projects using the FSM with ML-based potentials please cite:

Marks, J., & Gomes, J. (2025). Efficient Transition State Searches by Freezing String Method with Graph Neural Network Potentials. http://arxiv.org/abs/2501.06159

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Third-Party Licenses and Attribution

This project depends on several third-party open-source Python packages. These dependencies are not bundled with this repository and must be installed separately by the user.

Below is a list of direct dependencies and their respective licenses:

| Package             | License       | Link |
|---------------------|---------------|------|
| ASE                 | LGPL-2.1      | https://gitlab.com/ase/ase |
| geomeTRIC           | BSD 3-Clause  | https://github.com/leeping/geomeTRIC |
| NumPy               | BSD 3-Clause  | https://numpy.org/ |
| SciPy               | BSD 3-Clause  | https://scipy.org/ |
| NetworkX            | BSD 3-Clause  | https://networkx.org/ |

These licenses are all compatible with the MIT license under which this project is distributed. Please refer to each package’s own repository for the full license text.


## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
