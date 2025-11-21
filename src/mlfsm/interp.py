"""Interpolation methods for constructing paths between endpoint geometries."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from ase import Atoms
from ase.units import Bohr
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

from mlfsm.coords import Redundant

angs_to_bohr = 1 / Bohr
deg_to_rad = np.pi / 180.0


@dataclass
class Interpolate:
    """Abstract base class for interpolation schemes between molecular geometries."""

    atoms1: Atoms
    atoms2: Atoms
    ninterp: int
    gtol: float = 1e-4
    return_q: bool = False

    def interpolate(self) -> NDArray[np.float32]:
        """Abstract interpolationn routine-must be overridden."""
        raise NotImplementedError

    def __call__(self) -> NDArray[np.float32]:
        """Call the interpolation routine and returns interpolated geometries."""
        return self.interpolate()


class Linear(Interpolate):
    """Linear interpolation of Cartesian coordinates.

    Generates a reaction path by linearly interpolating Cartesian coordinates
    between two endpoint geometries.
    """

    def interpolate(self) -> NDArray[np.float32]:
        """Compute linear interpolated path between two geometries."""
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()

        def xab(f: float) -> NDArray[np.floating]:
            return np.array((1 - f) * xyz1 + f * xyz2, dtype=np.float64)

        fs = np.linspace(0, 1, self.ninterp)
        # build a 2D float32 array of shape (ninterp, natoms*3)
        return np.array([xab(f).flatten() for f in fs], dtype=np.float32)


class LST(Interpolate):
    """Linear Synchronous Transit (LST) interpolation method.

    Halgren, Thomas A., and William N. Lipscomb.
    "The synchronous transit method for determining reaction pathways and locating molecular transition states."
    Chemical Physics Letters 49.2 (1977): 225 to 232.
    """

    def obj(
        self,
        x_c: NDArray[np.floating],
        f: float,
        rab: Callable[[float], NDArray[np.floating]],
        xab: Callable[[float], NDArray[np.floating]],
    ) -> float:
        """Objective function for LST interpolation."""
        x_c = x_c.reshape(-1, 3)
        rab_c = pdist(x_c)
        rab_i = rab(f)
        x_i = xab(f).reshape(-1, 3)

        return float((((rab_i - rab_c) ** 2) / rab_i**4).sum() + 5e-2 * ((x_i - x_c) ** 2).sum())

    def interpolate(self) -> NDArray[np.float32]:
        """Generate interpolated structures using LST."""
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()
        pdist_1 = pdist(xyz1)
        pdist_2 = pdist(xyz2)

        def rab(f: float) -> NDArray[np.floating]:
            return np.array((1 - f) * pdist_1 + f * pdist_2, dtype=np.float64)

        def xab(f: float) -> NDArray[np.floating]:
            return np.array((1 - f) * xyz1 + f * xyz2, dtype=np.float64)

        minimize_kwargs = {
            "method": "L-BFGS-B",
            "options": {
                "gtol": self.gtol,
            },
        }
        string = [xab(0).flatten()]
        string += [
            minimize(self.obj, x0=xab(f).flatten(), args=(f, rab, xab), **minimize_kwargs).x  # type: ignore [call-overload]
            for f in np.linspace(0, 1, self.ninterp)[1:-1]
        ]
        string += [xab(1).flatten()]

        return np.array(string, dtype=np.float32)


@dataclass
class RIC(Interpolate):
    """Interpolates in redundant internal coordinates (RIC)."""

    def __post_init__(self) -> None:
        """Initialize the RIC interpolator."""
        self.coords = Redundant(self.atoms1, self.atoms2, verbose=False)

    def interpolate(self) -> NDArray[np.float32]:
        """Generate interpolated structures using linear interpolation in RIC."""
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()
        q1 = self.coords.q(xyz1)  # type ignore[no-untyped-call]
        q2 = self.coords.q(xyz2)  # type ignore[no-untyped-call]
        dq = q2 - q1
        for i, name in enumerate(self.coords.keys):
            if ("tors" in name) and dq[i] > np.pi:
                while q1[i] < np.pi:
                    q1[i] += 2 * np.pi
            elif ("tors" in name) and dq[i] < -np.pi:
                while q2[i] < np.pi:
                    q2[i] += 2 * np.pi

        def xab(f: float) -> NDArray[np.float64]:
            return (1 - f) * q1 + f * q2

        fs = np.linspace(0, 1, self.ninterp)
        if self.return_q:
            string = []
            for f in fs:
                string.append(xab(f))
            return np.array(string, dtype=np.float32)

        xyz = xyz1
        string = []
        for f in fs:
            xyz = self.coords.x(xyz, xab(f))  # type ignore[no-untyped-call]
            string.append(xyz)

        return np.array(string, dtype=np.float32)
