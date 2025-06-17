# FSM

This repository provides an implementation of the Freezing String Method (FSM) for double-ended transition state searches with internal coordinates interpolation.

## Installation

Clone the repository and install dependencies listed in `requirements.txt`:

```bash
git clone https://github.com/thegomeslab/fsm.git
cd fsm
pip install -r requirements.txt
```

## Example
For interactive use see FSM_GNN_Colab_Example.ipynb

To run the FSM with default parameters and on a Diels Alder reaction run:
```python
python fsm.py data/sharada/06_diels_alder/ 
```
Note: Users are responsible for installing their desired quantum chemistry backend, current calculators supported are SchNet, AIMNet2, MACEOFF23, FAIR UMA, TensorNet, xTB, QChem.

FSM parameters and configuration options can be adjusted directly in fsm.py

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
