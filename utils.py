import os
from ase.io import read, write
from geom import project_trans_rot

def load_xyz(reaction_dir):

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

