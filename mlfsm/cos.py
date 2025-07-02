"""Freezing String Method driver for reaction pathway construction."""

import os

import numpy as np
from scipy.interpolate import CubicSpline

from .coords import Cartesian
from .geom import (
    calculate_arc_length,
    distance,
    normalize,
    project_trans_rot,
)
from .interp import LST, RIC, Linear
from .utils import float_check


class FreezingString(object):
    """Implements the Freezing String Method."""

    def __init__(self, reactant, product, nnodes_min=10, interp="lst", ninterp=100):
        self.nnodes_min = int(nnodes_min)
        self.ninterp = int(ninterp)
        if interp == "cart":
            self.interp = Linear
        elif interp == "lst":
            self.interp = LST
        elif interp == "ric":
            self.interp = RIC
        else:
            raise Exception("Check interpolation method")

        self.atoms = reactant.copy()
        self.natoms = len(self.atoms.numbers)

        interp = self.interp(reactant, product, ninterp=self.ninterp)
        string = interp()
        s = calculate_arc_length(string)
        self.dist = s[-1]
        self.stepsize = self.dist / self.nnodes_min
        print(f"NNODES_MIN: {self.nnodes_min}")
        print(f"DIST: {self.dist:.3f} STEPSIZE: {self.stepsize:.3f}")

        self.r_string = [reactant.copy()]
        self.r_fix = [True]
        self.r_energy = [None]
        self.r_tangent = [None]
        self.r_nnodes = len(self.r_string)
        self.p_string = [product.copy()]
        self.p_fix = [True]
        self.p_energy = [None]
        self.p_tangent = [None]
        self.p_nnodes = len(self.p_string)

        self.growing = True
        self.iteration = 0
        self.ngrad = 0

        self.coordsobj = None

    def interpolate(self, outdir):
        """Generate and write interpolated string between current endpoints."""
        r_atoms = self.r_string[-1]
        p_atoms = self.p_string[-1]

        r_xyz, p_xyz = project_trans_rot(r_atoms.get_positions(), p_atoms.get_positions())
        r_xyz, p_xyz = r_xyz.flatten(), p_xyz.flatten()

        interp = self.interp(r_atoms, p_atoms, ninterp=self.ninterp)
        string = interp()
        s = calculate_arc_length(string)

        path = []
        for i in range(self.ninterp):
            atoms = self.atoms.copy()
            atoms.set_positions(string[i].reshape(-1, 3))
            path += [atoms.copy()]

        outfile = os.path.join(outdir, "interp_{}.xyz".format(str(self.iteration).zfill(2)))
        with open(outfile, "w") as f:
            for i, atoms in enumerate(path):
                f.write(f"{self.natoms}\n")
                f.write(f"{s[i]:.5f}\n")
                for atom, xyz in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                    f.write(f"{atom} {float(xyz[0]):.8f} {float(xyz[1]):.8f} {float(xyz[2]):.8f}\n")

    def grow(self):
        """Grow the string by adding nodes from each end."""
        r_atoms = self.r_string[-1]
        p_atoms = self.p_string[-1]

        r_xyz, p_xyz = project_trans_rot(r_atoms.get_positions(), p_atoms.get_positions())
        r_xyz, p_xyz = r_xyz.flatten(), p_xyz.flatten()

        interp = self.interp(r_atoms, p_atoms, ninterp=self.ninterp)
        try:
            self.coordsobj = interp.coords
        except Exception:
            self.coordsobj = Cartesian(r_atoms, p_atoms)
        string = interp()
        s = calculate_arc_length(string)
        cs = CubicSpline(s, string.reshape(self.ninterp, 3 * self.natoms), axis=0)
        self.dist = s[-1]

        if self.dist < self.stepsize:
            self.growing = False
            return

        r_idx = np.abs(s - self.stepsize).argmin()
        p_idx = np.abs(s - (s[-1] - self.stepsize)).argmin()
        r_frontier = self.atoms.copy()
        r_frontier.set_positions(string[r_idx].reshape(-1, 3))

        self.r_string += [r_frontier]
        self.r_fix += [False]
        self.r_energy += [None]
        self.r_tangent += [normalize(cs(s[r_idx], 1))]
        self.r_nnodes = len(self.r_string)

        if self.dist <= 2 * self.stepsize:
            self.growing = False
            return

        p_frontier = self.atoms.copy()
        p_frontier.set_positions(string[p_idx].reshape(-1, 3))

        self.p_string += [p_frontier]
        self.p_fix += [False]
        self.p_energy += [None]
        self.p_tangent += [normalize(cs(s[p_idx], 1))]
        self.p_nnodes = len(self.p_string)

    def optimize(self, optimizer):
        """Relax unfixed nodes on the hyperplane orthogonal to the local tangent direction."""
        self.iteration += 1
        optimizer.coordsobj = self.coordsobj

        for i in range(self.r_nnodes):
            if self.r_energy[i] is None and self.r_fix[i]:
                energy = optimizer.calc.get_potential_energy(self.r_string[i])
                self.r_energy[i] = float_check(energy)
            elif not self.r_fix[i]:
                assert self.r_tangent[i] is not None
                atoms = self.r_string[i]
                try:
                    atoms, energy, ngrad = optimizer.optimize(atoms, self.r_tangent[i])
                    self.r_string[i] = atoms
                    self.r_energy[i] = float_check(energy)
                except Exception:
                    energy = optimizer.calc.get_potential_energy(atoms)
                    self.r_energy[i] = float_check(energy)
                    ngrad = 0
                self.r_fix[i] = True
                self.ngrad += ngrad

        for i in range(self.p_nnodes):
            if self.p_energy[i] is None and self.p_fix[i]:
                energy = optimizer.calc.get_potential_energy(self.p_string[i])
                self.p_energy[i] = float_check(energy)
            elif not self.p_fix[i]:
                assert self.p_tangent[i] is not None
                atoms = self.p_string[i]
                try:
                    atoms, energy, ngrad = optimizer.optimize(atoms, self.p_tangent[i])
                    self.p_string[i] = atoms
                    self.p_energy[i] = float_check(energy)
                except Exception:
                    energy = optimizer.calc.get_potential_energy(atoms)
                    self.p_energy[i] = float_check(energy)
                    ngrad = 0
                self.p_fix[i] = True
                self.ngrad += ngrad

        self.dist = distance(self.r_string[-1].get_positions().flatten(), self.p_string[-1].get_positions().flatten())

        if self.dist < self.stepsize:
            self.growing = False

    def write(self, outdir):
        """Write current string geometries and relative energies to an XYZ file."""
        outfile = os.path.join(outdir, "vfile_{}.xyz".format(str(self.iteration).zfill(2)))
        path = self.r_string + self.p_string[::-1]
        string = np.stack([atoms.get_positions() for atoms in path], axis=0)
        s = calculate_arc_length(string)
        energy = np.array(self.r_energy + self.p_energy[::-1])
        energy = energy - energy.min()  # now will be in just eV
        with open(outfile, "w") as f:
            for i, atoms in enumerate(path):
                _, xyz = project_trans_rot(string[0], string[i])
                xyz = xyz.reshape(-1, 3)
                f.write(f"{self.natoms}\n")
                f.write(f"{s[i]:.5f} {energy[i]:.3f}\n")
                for atom, coord in zip(atoms.get_chemical_symbols(), xyz):
                    f.write(f"{atom} {float(coord[0]):.8f} {float(coord[1]):.8f} {float(coord[2]):.8f}\n")
        energy_str = np.array2string(energy, precision=1, floatmode="fixed")
        print(f"ITERATION: {self.iteration} DIST: {self.dist:.2f} ENERGY: {energy_str}")

        if not self.growing:
            gradfile = os.path.join(outdir, "ngrad.txt")
            with open(gradfile, "w") as f:
                f.write(f"{self.ngrad}\n")
