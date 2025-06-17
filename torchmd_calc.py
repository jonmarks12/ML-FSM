import os
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torchmdnet.models.model import load_model
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
from sella import Sella
from torch import Tensor

class TMDCalculator(Calculator):
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, **kwargs):
        
        Calculator.__init__(self, **kwargs)
        checkpoint = "/Users/jonmarks/ts_searches/fsm/gnns/epoch=359-val_loss=0.0212-test_loss=0.2853.ckpt"
        self.model = load_model(checkpoint, derivative=True)
        self.z = None
        self.batch = None
        
    def calculate(self, atoms=None, properties=['energy', 'forces'], 
                 system_changes=all_changes):
        
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
            'energy': energy,
            'forces': forces,
        }