from sympy import ImmutableMatrix, MatrixSymbol, Symbol
from pycalphad.core.cache import cacheit
from pycalphad.core.sympydiff_utils import AutowrapFunction, CompileLock
import pycalphad.variables as v


@cacheit
def build_constraint_functions(constraints, rhs, variables, parameters=None):
    """

    Parameters
    ----------
    constraints : list of SymPy objects
    rhs : list of SymPy symbols representing the "right-hand side" of the constraint equation
    variables : tuple of Symbols
        Input arguments.
    parameters

    Returns
    -------
    Constraint function, constraint Jacobian
    """
    wrt = tuple(variables)
    if parameters is None:
        parameters = []
    new_parameters = []
    for param in parameters:
        if isinstance(param, Symbol):
            new_parameters.append(param)
        else:
            new_parameters.append(Symbol(param))
    parameters = tuple(new_parameters)
    variables = tuple(variables)
    rhs = tuple(rhs)
    params = MatrixSymbol('params', 1, len(parameters))
    cons_rhs = MatrixSymbol('cons_rhs', 1, len(rhs))
    inp_nobroadcast = MatrixSymbol('inp', 1, len(variables))

    args_nobroadcast = []
    for indx in range(len(variables)):
        args_nobroadcast.append(inp_nobroadcast[0, indx])
    for indx in range(len(parameters)):
        args_nobroadcast.append(params[0, indx])
    for indx in range(len(cons_rhs)):
        args_nobroadcast.append(cons_rhs[0, indx])

    funcargs = (inp_nobroadcast, params, rhs)
    nobroadcast = dict(zip(variables+parameters+rhs, args_nobroadcast))
    sympy_graph_nobroadcast = [x.xreplace(nobroadcast) for x in constraints]
    cons_func = AutowrapFunction(funcargs, ImmutableMatrix(sympy_graph_nobroadcast))
    with CompileLock:
        cons_diffs = list([cons_nobroadcast.diff(nobroadcast[i]) for i in wrt]
                          for cons_nobroadcast in sympy_graph_nobroadcast)
    jac_func = AutowrapFunction(funcargs, ImmutableMatrix(cons_diffs))
    return cons_func, jac_func


class Constraints(object):
    def __init__(self, conditions, models, variables, parameters=None):
        parameters = parameters if parameters is not None else []
        constraints = []
        rhs = []
        # User-specified constraints
        for cond, value in conditions.items():
            constraints.append(cond.as_dof())
            rhs.append(value)
        # Sublattice site fraction balance constraints (mandatory)
        for model in models.values():
            for idx, sublattice in enumerate(model.constituents):
                active = set(sublattice).intersection(model.components)
                if len(active) > 0:
                    balance = sum(v.SiteFraction(model.phase_name, idx, spec) for spec in active)
                    constraints.append(balance)
                    rhs.append(1)
        self.symbols = constraints
        self.rhs = rhs
        self.parameters = parameters
        self.cons_func, self.jac_func = \
            build_constraint_functions(constraints,
                                       [Symbol('_RHS_{}'.format(i)) for i in range(len(rhs))], variables,
                                       parameters=parameters)
