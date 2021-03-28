import abc
import textwrap
from typing import Mapping, Sequence

import sympy

from pycalphad import variables as v

# Maximum number of levels deep we check for symbols that are functions of
# other symbols
_MAX_PARAM_NESTING = 32
MODEL_REGISTRY = []  # all ModelBase subclasses go in here

# TODO: document API
class ModelProtocol(abc.ABC):
    """
    ModelProtocol defines the minimal API required for an equilibrium calculation.

    Any class that implements the defined properties and methods correctly could
    be used for equilibrium calculations in pycalphad.

    The purpose of this class is to be able to guarantee (by static type
    checking) that calculate and equilibrium only use a subset of all this API.
    """
    # constituents should correspond to the active constituents in each
    # sublattice corresponding to how those constituents are represented in the
    # Gibbs energy model. Usually the species in Model.constituents are subsets
    # of Phase.constituents with the inactive species filtered out, but some
    # models can have internal virtual constituents that don't necessarily match
    # Phase.constituents and this allows pycalphad to handle this correctly.
    constituents: Sequence[Sequence[v.Species]]
    GM: sympy.Expr  # TODO: Units J/mol-atom
    site_fractions: [v.SiteFraction]
    state_variables: [v.StateVariable]

    # TODO: When is this used for pure element species and when for other
    #       species? Does there need to be a distinction in the API so
    #       subclassers know that these cases (pure element moles vs species
    #       moles) need to be handled differently?
    @abc.abstractmethod
    def moles(self, species: v.Species) -> sympy.Expr:
        """Return an expression to calculate the moles of a pure element"""

    @abc.abstractmethod
    def get_internal_constraints(self) -> [sympy.Expr]:
        """Return a list of expressions for the internal constraints of this phase

        Usually the internal constraints correspond to constraints that should
        be valid within the phase internal degrees of freedom. Common examples
        are that the sum of site fractions on a sublattice should sum to unity
        or that the total charge is neutral.

        When each constraint is satisfied, the constraint expressions should
        evaluate to zero.
        """

    # TODO: Better comment on what multiphase constraints are expected and how
    #       they interact with each other.
    @abc.abstractmethod
    def get_multiphase_constraints(self, conds) -> [sympy.Expr]:
        """Return a list of expressions for the internal constraints of this phase

        When each constraint is satisfied, the constraint expressions should
        evaluate to zero.
        """


class ModelBase(ModelProtocol):
    GM: sympy.Expr = property(lambda self: self.ast)
    energy: sympy.Expr = GM
    models: Mapping[str, sympy.Expr]

    """Base class defining the protocol for Gibbs energy models"""
    def __init_subclass__(cls, **kwargs):
        if cls not in MODEL_REGISTRY:
            MODEL_REGISTRY.append(cls)

    @abc.abstractmethod
    def __init__(self, dbf, comps, phase_name, parameters=None):
        """"""

    @staticmethod
    def dispatches_on(phase_obj: 'Phase') -> bool:
        # By default, all classes
        return True

    # Default methods, don't need to be overridden unless you want to change the behavior
    @property
    def state_variables(self) -> [v.StateVariable]:
        """Return a sorted list of state variables used in the ast which are not site fractions."""
        return sorted((x for x in self.ast.free_symbols if not isinstance(x, v.SiteFraction) and isinstance(x, v.StateVariable)), key=str)

    @property
    def site_fractions(self) -> [v.SiteFraction]:
        """Return a sorted list of site fractions used in the ast."""
        return sorted((x for x in self.ast.free_symbols if isinstance(x, v.SiteFraction)), key=str)

    @property
    def ast(self):
        "Return the full abstract syntax tree of the model."
        return sum(self.models.values())


class Model:
    def __new__(cls, dbf, comps, phase_name, parameters=None):
        # Reverse order so that the last model registered (likely a user model)
        # is the first to be dispatched on.
        for _Model in reversed(MODEL_REGISTRY):
            if _Model.dispatches_on(dbf.phases[phase_name.upper()]):
                mod = _Model(dbf, comps, phase_name, parameters=parameters)
                return mod

    def __init_subclass__(cls, **kwargs):
        msg = """
        Model objects cannot be subclassed directly.
        Custom Gibbs energy models should subclass ModelBase.
        """
        msg = ' '.join(textwrap.dedent(msg).splitlines())
        raise TypeError(msg)

    # Credit: https://stackoverflow.com/a/31075641
    @staticmethod
    def extend_instance(obj, cls):
        base_cls = obj.__class__
        base_cls_name = obj.__class__.__name__
        obj.__class__ = type(base_cls_name, (base_cls, cls), {})
