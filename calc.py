import os
import glob
import numpy as np
import subprocess
from tempfile import NamedTemporaryFile
from ase.units import Bohr

class XTBRunner(object):

    def __init__(self, atoms, charge=0, mult=1, nt=12):

        self.atoms = atoms
        self.numbers = atoms.get_atomic_numbers()
        tf = NamedTemporaryFile(mode='w', suffix='.xyz', delete=False)
        self.filename = tf.name
        tf.close()

        self.charge = charge
        self.mult = mult
        self.nt = nt

    def write_xyz(self, positions):

        n_atoms = len(self.numbers)
        positions = positions.reshape(n_atoms, 3)
        with open(self.filename, 'w') as f:
            f.write(f'{n_atoms}\n\n')
            for ix, ixyz in zip(self.numbers, positions):
                f.write("{} {} {} {}\n".format(ix, *ixyz))
        return self.filename
  
    def energy(self, positions):

        filename = self.write_xyz(positions)
        runcmd = ['xtb', filename, '--ceasefiles', '--T', str(self.nt)]
        if self.charge != 0:
            runcmd += ['--chrg', str(self.charge)]
        if self.mult != 1:
            runcmd += ['--uhf', str(self.mult)]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        for line in reversed(proc.stdout.split("\n")):
            if "TOTAL ENERGY" in line:
                try:
                    energy = float(line.split()[3])
                    return energy
                except:
                    return float("nan")

        for f in ['energy', 'gradient'] + list(glob.glob("*.engrad")):
            if os.path.exists(f): os.unlink(f)
      
        return float("nan")

    def grad(self, positions):
        filename = self.write_xyz(positions)
        runcmd = ['xtb', filename, '--ceasefiles', '--grad', '--T', str(self.nt)]
        if self.charge != 0:
            runcmd += ['--chrg', str(self.charge)]
        if self.mult != 1:
            runcmd += ['--uhf', str(self.mult)]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        with open('gradient', 'r') as f:
            data = f.readlines()
        n_atoms = len(self.numbers)
        energy = float(data[1].split()[6])
        grad = np.array([[float(i) for i in l.split()] for l in data[n_atoms+2:-1]]).reshape(-1)
        # convert grad from Hartree/Bohr to Hartree/Angs
        grad /= Bohr

        for f in ['energy', 'gradient'] + list(glob.glob("*.engrad")):
            if os.path.exists(f): os.unlink(f)

        return energy, grad
    
    def hess(self, positions):
        filename = self.write_xyz(positions)
        runcmd = ['xtb', filename, '--ceasefiles', '--hess', '--T', str(self.nt)]
        if self.charge != 0:
            runcmd += ['--chrg', str(self.charge)]
        if self.mult != 1:
            runcmd += ['--uhf', str(self.mult)]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        natoms = len(self.numbers)
        with open('hessian', 'r') as f:
            data = f.readlines()
        hess = [[float(i) for i in l.split()] for l in data[1:]]
        hess = []
        for l in data[1:]:
            hess += [float(i) for i in l.split()]
        hess = np.array(hess).reshape(3*natoms, 3*natoms)
        #print(proc.stdout)
        #convert units?
        return hess
    
    def __call__(self, positions):
        return self.grad(positions)


class QChemRunner(object):

    def __init__(self, atoms, charge=0, mult=1, nt=16):

        self.atoms = atoms
        self.numbers = atoms.get_atomic_numbers()
        tf = NamedTemporaryFile(mode='w', suffix='.qcin', delete=False)
        self.filename = tf.name
        tf.close()

        self.charge = charge
        self.mult = mult
        self.nt = nt

    def write(self, positions, jobtype='sp'):

        positions = positions.reshape(-1, 3)
        rem = {'jobtype': jobtype,
                'method': 'wb97x-v',
                'basis': 'def2-tzvp',
                'sym_ignore': 'true',
                'symmetry': 'false',
                'scf_algorithm': 'diis_gdm',
                'scf_max_cycles': '500'}
        if jobtype == 'freq':
            rem['vibman_print'] = '4'

        with open(self.filename, 'w') as f:
            f.write(f"$molecule\n{self.charge} {self.mult}\n")
            [f.write("{} {} {} {}\n".format(ix, *ixyz)) for ix, ixyz in zip(self.numbers, positions)]
            f.write("$end\n\n$rem\n")
            [f.write(f"{k} {v}\n") for k,v in rem.items()]
            f.write("$end\n")
      
        return self.filename
  
    def energy(self, positions):

        filename = self.write(positions, jobtype='sp')
        runcmd = ['qchem', '-nt', str(self.nt), filename]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        #print(proc.stdout)
        energy = None
        for line in proc.stdout.split("\n"):
            if "Total energy in the final basis set" in line:
                energy = float(line.split()[-1])
        return energy

    def grad(self, positions):

        filename = self.write(positions, jobtype='force')
        runcmd = ['qchem', '-nt', str(self.nt), filename]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        #print(proc.stdout)
        energy = None
        for line in proc.stdout.split("\n"):
            if "Total energy in the final basis set" in line:
                energy = float(line.split()[-1])

        natoms = len(self.atoms)
        nblocks = natoms // 6
        if natoms%6 > 0: nblocks += 1
        grad = np.zeros((natoms, 3))
        out = iter(proc.stdout.split("\n"))
        for line in out:
            if "Gradient of SCF Energy" in line:
                for block in range(nblocks):
                    next(out)
                    for i in range(3):
                        line = next(out)[5:].rstrip()
                        grad_i = np.array([float(line[j:j+12]) for j in range(0, len(line), 12)])
                        grad[6*block:6*block+len(grad_i), i] = grad_i
                grad = (grad/Bohr).reshape(-1)
                break
        return energy, grad
    
    def hess(self, positions):
        filename = self.write(positions, jobtype='freq')
        runcmd = ['qchem', '-nt', str(self.nt), filename]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        data = proc.stdout.split("\n")
        out = iter(data)
        for line in out:
            if "$molecule" in line:
                next(out)
                natoms = 0
                while True:
                    l = next(out)
                    if "$end" in l: break
                    natoms += 1
                break

        nblocks = 3*natoms // 6
        if (3*natoms)%6 > 0: nblocks += 1
        hess = np.zeros((3*natoms, 3*natoms))
        for line in out:
            if "Mass-Weighted Hessian Matrix:" in line:
                for block in range(nblocks):
                    next(out)
                    next(out)
                    for i in range(3*natoms):
                        hline = next(out)[4:].rstrip()
                        hess_i = np.array([float(hline[j:j+12]) for j in range(0, len(hline), 12)])
                        hess[i, 6*block:6*block+len(hess_i)] = hess_i
                break
        return hess
    
    def __call__(self, positions):
        return self.grad(positions)

