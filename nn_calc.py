import os
import glob
import numpy as np
import subprocess
from tempfile import NamedTemporaryFile
from ase.units import Bohr

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch_geometric.nn import SchNet
from torch import Tensor
import numpy as np
from ase import Atoms
import torch
from torchmetrics import MeanSquaredError
from torch.autograd import grad

class SchNetLightning(LightningModule):
    def __init__(self, hidden_channels: int, num_filters: int,
                  num_interactions: int, num_gaussians: int,
                  cutoff: float):

        super().__init__()
        self.save_hyperparameters()

        self.model = SchNet(hidden_channels=hidden_channels,
                        num_filters=num_filters,
                        num_interactions=num_interactions,
                        num_gaussians=num_gaussians,
                        cutoff=cutoff)

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        return self.model(x, pos)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)

class SchNetRunner(object):

    def __init__(self,reactant,checkpoint='gnns/schnet_fine_tuned.ckpt'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.make_atoms_object(reactant)
        self.create_model(checkpoint)

    def make_atoms_object(self,reactant):
        pos = np.array(reactant.get_positions())
        self.atoms = Atoms(reactant.get_chemical_symbols(), positions = pos)


    def create_model(self,checkpoint):
        self.model = SchNetLightning.load_from_checkpoint(checkpoint,strict=False).to(self.device)
        print('Pre-trained model parameters loaded')
        
    def energy(self,x):
        self.model.eval()
        pos = torch.tensor(x.reshape((-1,3))).to(self.device).float()
        self.atoms.set_positions(x.reshape(-1,3))
        z = torch.tensor(self.atoms.get_atomic_numbers(),dtype=float).long().to(self.device)
        energy = self.model(z,pos)
        return energy.detach().cpu().numpy()[0][0]


    def grad(self,x):
        self.model.eval()
        pos = torch.tensor(x.reshape((-1,3)),requires_grad=True).to(self.device).float()
        self.atoms.set_positions(x.reshape(-1,3))
        z = torch.tensor(self.atoms.get_atomic_numbers(),requires_grad=True,dtype=float).long().to(self.device)
        energy = self.model(z,pos)
        x_grad = grad(energy, pos)[0]
        force = x_grad.detach().cpu().numpy().flatten()
        return energy.detach().cpu().numpy()[0][0],force

    def __call__(self,x):
        return self.grad(x)


if __name__ == '__main__':

    from ase import Atoms
    ckpt = "schnet_fine_tuned.ckpt"
    d = 1.1
    atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)])
    calculator = SchNetRunner(atoms, ckpt)

