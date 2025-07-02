"""Interpolation methods for constructing paths between endpoint geometries."""

import numpy as np
from ase.units import Bohr
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

from .coords import Redundant

angs_to_bohr = 1 / Bohr
deg_to_rad = np.pi / 180.0


class Interpolate(object):
    """Abstract base class for interpolation schemes between molecular geometries."""

    def __init__(self, atoms1, atoms2, ninterp, gtol=1e-4):
        self.atoms1 = atoms1
        self.atoms2 = atoms2
        self.ninterp = ninterp
        self.gtol = gtol

    def __call__(self):
        """Call the interpolation routine and returns interpolated geometries."""
        return self.interpolate()


class Linear(Interpolate):
    """Linear interpolation of Cartesian coordinates.

    Generates a reaction path by linearly interpolating Cartesian coordinates
    between two endpoint geometries.
    """

    def interpolate(self):
        """Compute linear interpolated path between two geometries."""
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()

        def xab(f):
            return (1 - f) * xyz1 + f * xyz2

        string = []
        fs = np.linspace(0, 1, self.ninterp)
        for _, f in enumerate(fs):
            x0 = xab(f).flatten()
            string.append(x0)
        return np.array(string, dtype=np.float32)


class LST(Interpolate):
    """Linear Synchronous Transit (LST) interpolation method.

    Halgren, Thomas A., and William N. Lipscomb.
    "The synchronous transit method for determining reaction pathways and locating molecular transition states."
    Chemical Physics Letters 49.2 (1977): 225 to 232.
    """

    def obj(self, x_c, f, rab, xab):
        """Objective function for LST interpolation."""
        x_c = x_c.reshape(-1, 3)
        rab_c = pdist(x_c)
        rab_i = rab(f)
        x_i = xab(f).reshape(-1, 3)
        return (((rab_i - rab_c) ** 2) / rab_i**4).sum() + 5e-2 * ((x_i - x_c) ** 2).sum()

    def interpolate(self):
        """Generate interpolated structures using LST."""
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()
        pdist_1 = pdist(xyz1)
        pdist_2 = pdist(xyz2)

        def rab(f):
            return (1 - f) * pdist_1 + f * pdist_2

        def xab(f):
            return (1 - f) * xyz1 + f * xyz2

        minimize_kwargs = {
            "method": "L-BFGS-B",
            "options": {
                "gtol": self.gtol,
            },
        }
        string = [xab(0).flatten()]
        fs = np.linspace(0, 1, self.ninterp)[1:-1]
        for _, f in enumerate(fs):
            x0 = xab(f).flatten()
            res = minimize(self.obj, x0=x0, args=(f, rab, xab), **minimize_kwargs)
            string.append(res.x)
        string += [xab(1).flatten()]
        return np.array(string, dtype=np.float32)


class RIC(Interpolate):
    """Interpolates in redundant internal coordinates (RIC)."""

    def __init__(self, atoms1, atoms2, ninterp, gtol=1e-4):
        super().__init__(atoms1, atoms2, ninterp, gtol)
        self.coords = Redundant(atoms1, atoms2, verbose=False)
        # self.coords = Cartesian(atoms1, atoms2)

    def interpolate(self):
        """Generate interpolated structures using linear interpolation in RIC."""
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()
        q1 = self.coords.q(xyz1)
        q2 = self.coords.q(xyz2)
        dq = q2 - q1
        for i, name in enumerate(self.coords.keys):
            if ("tors" in name) and dq[i] > np.pi:
                # print(name, dq[i], q1[i])
                while q1[i] < np.pi:
                    q1[i] += 2 * np.pi
            elif ("tors" in name) and dq[i] < -np.pi:
                # print(name, dq[i], q2[i])
                while q2[i] < np.pi:
                    q2[i] += 2 * np.pi

        def xab(f):
            return (1 - f) * q1 + f * q2

        string = []
        fs = np.linspace(0, 1, self.ninterp)
        xyzref = xyz1.copy()
        for _, f in enumerate(fs):
            qtarget = xab(f)
            xyz = self.coords.x(xyzref, qtarget)
            string.append(xyz)
            xyzref = xyz.copy()

        return np.array(string, dtype=np.float32)
