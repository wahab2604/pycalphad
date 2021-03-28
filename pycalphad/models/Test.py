import numpy as np
from sympy import Add, sin
from pycalphad import variables as v
from base import ModelBase


class TestModel(ModelBase):
    """
    Test Model object for global minimization.

    Equation 15.2 in:
    P.M. Pardalos, H.E. Romeijn (Eds.), Handbook of Global Optimization,
    vol. 2. Kluwer Academic Publishers, Boston/Dordrecht/London (2002)

    Parameters
    ----------
    dbf : Database
        Ignored by TestModel but retained for API compatibility.
    comps : sequence
        Names of components to consider in the calculation.
    phase : str
        Name of phase model to build.
    solution : sequence, optional
        Float array locating the true minimum. Same length as 'comps'.
        If not specified, randomly generated and saved to self.solution

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.
    """

    def __init__(self, dbf, comps, phase, solution=None, kmax=None):
        self.components = set(comps)
        if 'VA' in self.components:
            raise ValueError('Vacancies are unsupported in TestModel')
        self.models = dict()
        variables = [v.SiteFraction(phase.upper(), 0, x) for x in sorted(self.components)]
        if solution is None:
            solution = np.random.dirichlet(np.ones_like(variables, dtype=np.int))
        self.solution = dict(list(zip(variables, solution)))
        kmax = kmax if kmax is not None else 2
        scale_factor = 1e4 * len(self.components)
        ampl_scale = 1e3 * np.ones(kmax, dtype=np.float)
        freq_scale = 10 * np.ones(kmax, dtype=np.float)
        polys = Add(*[ampl_scale[i] * sin(freq_scale[i] * Add(*[Add(*[(varname - sol)**(j+1)
                                                                      for varname, sol in self.solution.items()])
                                                                for j in range(kmax)]))**2
                      for i in range(kmax)])
        self.models['test'] = scale_factor * Add(*[(varname - sol)**2 for varname, sol in self.solution.items()]) + polys
