"""TorchMD-Net-based ASE calculator for ML-FSM."""

from typing import ClassVar

import torch
from ase.calculators.calculator import Calculator, all_changes
from torchmdnet.models.model import load_model  # type: ignore [import-not-found]


class TMDCalculator(Calculator):
    """ASE-compatible calculator using a pretrained TorchMD-Net model."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]  # type: ignore [misc]

    def __init__(self, **kwargs):
        """Initialize the calculator and load the TorchMD-Net model."""
        Calculator.__init__(self, **kwargs)
        checkpoint = "/Users/jonmarks/ts_searches/fsm/gnns/epoch=359-val_loss=0.0212-test_loss=0.2853.ckpt"
        self.model = load_model(checkpoint, derivative=True, remove_ref_energy=False)
        self.z = None
        self.batch = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Compute energy and forces for the given atoms using TorchMD-Net.

        Parameters
        ----------
        atoms : ASE Atoms object
            The molecular structure to evaluate.
        properties : list of str, optional
            Desired properties (default: ["energy", "forces"]).
        system_changes : list, optional
            System changes triggering recalculation.
        """
        if properties is None:
            properties = ["energy", "forces"]
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.pos = torch.from_numpy(positions).float().reshape(-1, 3)

        if self.z is None:
            self.z = torch.from_numpy(atoms.numbers).long()
            self.batch = torch.zeros(len(atoms.numbers), dtype=torch.long)

        energy, forces = self.model(self.z, self.pos, self.batch)
        energy = energy.item()
        forces = forces.detach().numpy()
        self.results = {
            "energy": energy,
            "forces": forces,
        }
