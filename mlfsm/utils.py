"""Utility functions for reading input and checking numeric types."""

import os

import numpy as np
from ase import Atoms
from ase.io import read

from mlfsm.geom import project_trans_rot


def load_xyz(reaction_dir: str) -> tuple[Atoms, Atoms]:
    """
    Load reactant and product geometries from a directory.

    Assumes a file named initial.xyz containing the reactant and product geometries.
    """
    xyz = os.path.join(reaction_dir, "initial.xyz")
    if not os.path.exists(xyz):
        raise Exception(f"Input file {xyz} not found.")
    atoms = read(xyz, format="xyz", index=":")
    reactant = atoms[0]
    product = atoms[-1]
    r_xyz, p_xyz = project_trans_rot(reactant.get_positions(), product.get_positions())
    reactant.set_positions(r_xyz.reshape(-1, 3))
    product.set_positions(p_xyz.reshape(-1, 3))
    return reactant, product


def float_check(x: float) -> float:
    """
    Convert scalars, 0D arrays, or single element containers to a float.

    Leaves floats alone. Raises an error for anything else.
    """
    if isinstance(x, float):
        return x
    elif isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
        return float(x[0])
    else:
        raise TypeError(f"Cannot convert safely to float: {x} ({type(x)})")
