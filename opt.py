from scipy.optimize import minimize
from coords import Cartesian, Redundant
from geom import *


class Optimizer(object):

    def __init__(self, calc, method='L-BFGS-B', maxiter=3, maxls=6, dmax=0.3):
        self.calc = calc
        self.method = method
        self.maxiter = int(maxiter)
        self.maxls = int(maxls)
        self.dmax = float(dmax)

    def obj(self, xyz, tangent):
        raise NotImplementedError("No objective function")

    def optimize(self, xyz, tangent):
        raise NotImplementedError("No optimize function")

class CartesianOptimizer(Optimizer):

    def obj(self, xyz, tangent):
        proj = generate_project_rt_tan(xyz.reshape(-1, 3), tangent)
        energy, grads = self.calc(xyz)
        pgrads = proj @ grads
        return energy, pgrads

    def optimize(self, atoms, tangent):
        xyz = atoms.get_positions().flatten()
        config = {'fun': self.obj,
                  'x0': xyz,
                  'args': (tangent,),
                  'jac': True,
                  'method': self.method,
                  'bounds': [[j-self.dmax, j+self.dmax] for j in xyz],
                  'options': {'maxiter': self.maxiter, 
                              'maxls': self.maxls,} 
                  }
        res = minimize(**config)
        atomsf = atoms.copy()
        atomsf.set_positions(res.x.reshape(-1, 3))
        return atomsf, res.fun, res.njev

class InternalsOptimizer(Optimizer):

    def __init__(self, calc, method='L-BFGS-B', maxiter=3, maxls=6, dmax=0.05):
        super().__init__(calc, method, maxiter, maxls, dmax)
        self.calc = calc
        self.method = method
        self.maxiter = int(maxiter)
        self.maxls = int(maxls)
        self.dmax = float(dmax)
        self.coords = Redundant
        #self.coords = Cartesian
        self.coordsobj = None
        self.angle_dmax = dmax * 1.

    def compute_bounds(self, q):
        bounds = []
        for i, k in enumerate(self.coordsobj.keys):
          if "linearbnd" in k:
            angle_min = max(-np.pi, q[i]-self.angle_dmax)
            angle_max = min(np.pi, q[i]+self.angle_dmax)
            bounds += [(angle_min, angle_max)]
          elif "bend" in k:
            angle_min = max(0, q[i]-self.angle_dmax)
            angle_max = min(np.pi, q[i]+self.angle_dmax)
            bounds += [(angle_min, angle_max)]
          elif "tors" in k:
            angle_min = max(-np.pi, q[i]-self.angle_dmax)
            angle_max = min(np.pi, q[i]+self.angle_dmax)
            bounds += [(angle_min, angle_max)]
          elif "oop" in k:
            angle_min = max(-np.pi, q[i]-self.angle_dmax)
            angle_max = min(np.pi, q[i]+self.angle_dmax)
            bounds += [(angle_min, angle_max)]
          else:
            bounds += [(q[i]-self.dmax, q[i]+self.dmax)]
        return bounds

    def obj(self, q, xyzref, tangent):
        xyz = self.coordsobj.x(xyzref, q)
        proj = generate_project_rt_tan(xyz.reshape(-1, 3), tangent)
        energy, grads = self.calc(xyz)
        pgrads = proj @ grads
        B = self.coordsobj.b_matrix(xyz)
        B_inv = np.linalg.pinv(B)
        BT_inv = np.linalg.pinv(B.T)
        P = B@B_inv
        pgrads = P@BT_inv@pgrads 
        return energy, pgrads

    def optimize(self, atoms, tangent):
        #self.coordsobj = self.coords(atoms, verbose=False)
        q = self.coordsobj.q(atoms.get_positions())
        xyz = atoms.get_positions()
        config = {'fun': self.obj,
                  'x0': q,
                  'args': (xyz, tangent,),
                  'jac': True,
                  'method': self.method,
                  'bounds': self.compute_bounds(q),
                  'options': {'maxiter': self.maxiter, 
                              'maxls': self.maxls, 
                              'iprint': 0},
                  }
        res = minimize(**config)
        xf = self.coordsobj.x(xyz, res.x)
        atomsf = atoms.copy()
        atomsf.set_positions(xf)
        return atomsf, res.fun, res.njev
