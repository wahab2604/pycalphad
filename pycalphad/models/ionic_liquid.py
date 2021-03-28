from .rkm import ModelRedlichKisterMuggianu

"""
The model module provides support for using a Database to perform
calculations under specified conditions.
"""
import copy
from sympy import exp, log, Abs, Add, And, Float, Mul, Piecewise, Pow, S, sin, StrictGreaterThan, Symbol, zoo, oo, nan
from tinydb import where
import pycalphad.variables as v
from pycalphad.core.errors import DofError
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.utils import unpack_components, get_pure_elements, wrap_symbol
from pycalphad.core.constraints import is_multiphase_constraint
from collections import OrderedDict

# Maximum number of levels deep we check for symbols that are functions of
# other symbols
_MAX_PARAM_NESTING = 32


class ModelIonicLiquid2SL(ModelRedlichKisterMuggianu):
    """
    Models use an abstract representation of the function
    for calculation of values under specified conditions.

    Parameters
    ----------
    dbe : Database
        Database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phase_name : str
        Name of phase model to build.
    parameters : dict or list
        Optional dictionary of parameters to be substituted in the model.
        A list of parameters will cause those symbols to remain symbolic.
        This will overwrite parameters specified in the database

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.

    Notes
    -----
    The two sublattice ionic liquid model has several special cases compared to
    typical models within the compound energy formalism. A few key differences
    arise. First, variable site ratios (modulated by vacancy site fractions)
    are used to charge balance the phase. Second, endmembers for neutral
    species and interactions among only neutral species should be specified
    using only one sublattice (dropping the cation sublattice). For
    understanding the special cases used throughout this class, users are
    referred to:
    Sundman, "Modification of the two-sublattice model for liquids",
    Calphad 15(2) (1991) 109-119 https://doi.org/d3jppb


    """

    @staticmethod
    def dispatches_on(phase_obj: 'Phase'):
        return phase_obj.model_hints.get('ionic_liquid_2SL', False)

    def __init__(self, dbe, comps, phase_name, parameters=None):
        self._dbe = dbe
        self._reference_model = None
        self.components = set()
        self.constituents = []
        self.phase_name = phase_name.upper()
        phase = dbe.phases[self.phase_name]
        self.site_ratios = list(phase.sublattices)
        active_species = unpack_components(dbe, comps)
        if len(phase.constituents) != 2:
            raise ValueError(f"Two-sublattice ionic liquid model must have two sublattices (got {len(phase.constituents)}).")
        for idx, sublattice in enumerate(phase.constituents):
            subl_comps = set(sublattice).intersection(active_species)
            self.components |= subl_comps
            # Support for variable site ratios in ionic liquid model
            if idx == 0:
                subl_idx = 1
            elif idx == 1:
                subl_idx = 0
            self.site_ratios[subl_idx] = Add(*[v.SiteFraction(self.phase_name, idx, spec) * abs(spec.charge) for spec in subl_comps])
        # Special treatment of "neutral" vacancies in 2SL ionic liquid
        # These are treated as having variable valence
        for idx, sublattice in enumerate(phase.constituents):
            subl_comps = set(sublattice).intersection(active_species)
            if v.Species('VA') in subl_comps:
                if idx == 0:
                    subl_idx = 1
                elif idx == 1:
                    subl_idx = 0
                self.site_ratios[subl_idx] += self.site_ratios[idx] * v.SiteFraction(self.phase_name, idx, v.Species('VA'))
        self.site_ratios = tuple(self.site_ratios)

        # Verify that this phase is still possible to build
        is_pure_VA = set()
        for sublattice in phase.constituents:
            sublattice_comps = set(sublattice).intersection(self.components)
            if len(sublattice_comps) == 0:
                # None of the components in a sublattice are active
                # We cannot build a model of this phase
                raise DofError(
                    '{0}: Sublattice {1} of {2} has no components in {3}' \
                    .format(self.phase_name, sublattice,
                            phase.constituents,
                            self.components))
            is_pure_VA.add(sum(set(map(lambda s : getattr(s, 'number_of_atoms'),sublattice_comps))))
            self.constituents.append(sublattice_comps)
        if sum(is_pure_VA) == 0:
            #The only possible component in a sublattice is vacancy
            #We cannot build a model of this phase
            raise DofError(
                '{0}: Sublattices of {1} contains only VA (VACUUM) constituents' \
                .format(self.phase_name, phase.constituents))
        self.components = sorted(self.components)
        desired_active_pure_elements = [list(x.constituents.keys()) for x in self.components]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements
                                        for el in constituents]
        self.pure_elements = sorted(set(desired_active_pure_elements))
        self.nonvacant_elements = [x for x in self.pure_elements if x != 'VA']

        # Convert string symbol names to sympy Symbol objects
        # This makes xreplace work with the symbols dict
        symbols = {Symbol(s): val for s, val in dbe.symbols.items()}

        if parameters is not None:
            self._parameters_arg = parameters
            if isinstance(parameters, dict):
                symbols.update([(wrap_symbol(s), val) for s, val in parameters.items()])
            else:
                # Lists of symbols that should remain symbolic
                for s in parameters:
                    symbols.pop(wrap_symbol(s))
        else:
            self._parameters_arg = None

        self._symbols = {wrap_symbol(key): value for key, value in symbols.items()}

        self.models = OrderedDict()
        self.build_phase(dbe)

        for name, value in self.models.items():
            self.models[name] = self.symbol_replace(value, symbols)

    def _array_validity(self, constituent_array):
        """
        Return True if the constituent_array contains only active species of the current Model instance.
        """
        # Allow an exception for the ionic liquid model, where neutral
        # species can be specified in the anion sublattice without any
        # species in the cation sublattice.
        if len(constituent_array) == 1:
            param_sublattice = constituent_array[0]
            model_anion_sublattice = self.constituents[1]
            if (set(param_sublattice).issubset(model_anion_sublattice) or (param_sublattice[0] == v.Species('*'))):
                return True
        return super()._array_validity(constituent_array)

    def redlich_kister_sum(self, phase, param_search, param_query):
        """
        Construct parameter in Redlich-Kister polynomial basis, using
        the Muggianu ternary parameter extension.
        """
        rk_terms = []

        # search for desired parameters
        params = param_search(param_query)
        for param in params:
            # iterate over every sublattice
            mixing_term = S.One
            for subl_index, comps in enumerate(param['constituent_array']):
                comp_symbols = None
                # convert strings to symbols
                if comps[0] == v.Species('*'):
                    # Handle wildcards in constituent array
                    comp_symbols = \
                        [
                            v.SiteFraction(phase.name, subl_index, comp)
                            for comp in sorted(set(phase.constituents[subl_index])\
                                .intersection(self.components))
                        ]
                    mixing_term *= Add(*comp_symbols)
                else:
                    if (
                        len(param['constituent_array']) == 1 and  # There's only one sublattice
                        all(const.charge == 0 for const in param['constituent_array'][0])  # All constituents are neutral
                    ):
                        # The constituent array is all neutral anion species in what would be the
                        # second sublattice. TDB syntax allows for specifying neutral species with
                        # one sublattice model. Set the sublattice index to 1 for the purpose of
                        # site fractions.
                        subl_index = 1
                    comp_symbols = \
                        [
                            v.SiteFraction(phase.name, subl_index, comp)
                            for comp in comps
                        ]
                    # We need to special case sorting for this model, because the constituents
                    # should not be alphabetically sorted. The model should be (C)(A, Va, B)
                    # for cations (C), anions (A), vacancies (Va) and neutrals (B). Thus the
                    # second sublattice should be sorted by species with charge, then by
                    # vacancies, if present, then by neutrals. Hint: in Thermo-Calc, using
                    # `set-start-constitution` for a phase will prompt you to enter site
                    # fractions for species in the order they are sorted internally within
                    # Thermo-Calc. This can be used to verify sorting behavior.

                    # Assume that the constituent array is already in sorted order
                    # alphabetically, so we need to rearrange the species first by charged
                    # species, then VA, then netural species. Since the cation sublattice
                    # should only have charged species by definition, this is equivalent to
                    # a no-op for the first sublattice.
                    charged_symbols = [sitefrac for sitefrac in comp_symbols if sitefrac.species.charge != 0 and sitefrac.species.number_of_atoms > 0]
                    va_symbols = [sitefrac for sitefrac in comp_symbols if sitefrac.species == v.Species('VA')]
                    neutral_symbols = [sitefrac for sitefrac in comp_symbols if sitefrac.species.charge == 0 and sitefrac.species.number_of_atoms > 0]
                    comp_symbols = charged_symbols + va_symbols + neutral_symbols

                    mixing_term *= Mul(*comp_symbols)
                # is this a higher-order interaction parameter?
                if len(comps) == 2 and param['parameter_order'] > 0:
                    # interacting sublattice, add the interaction polynomial
                    mixing_term *= Pow(comp_symbols[0] - \
                        comp_symbols[1], param['parameter_order'])
                if len(comps) == 3:
                    # 'parameter_order' is an index to a variable when
                    # we are in the ternary interaction parameter case

                    # NOTE: The commercial software packages seem to have
                    # a "feature" where, if only the zeroth
                    # parameter_order term of a ternary parameter is specified,
                    # the other two terms are automatically generated in order
                    # to make the parameter symmetric.
                    # In other words, specifying only this parameter:
                    # PARAMETER G(FCC_A1,AL,CR,NI;0) 298.15  +30300; 6000 N !
                    # Actually implies:
                    # PARAMETER G(FCC_A1,AL,CR,NI;0) 298.15  +30300; 6000 N !
                    # PARAMETER G(FCC_A1,AL,CR,NI;1) 298.15  +30300; 6000 N !
                    # PARAMETER G(FCC_A1,AL,CR,NI;2) 298.15  +30300; 6000 N !
                    #
                    # If either 1 or 2 is specified, no implicit parameters are
                    # generated.
                    # We need to handle this case.
                    if param['parameter_order'] == 0:
                        # are _any_ of the other parameter_orders specified?
                        ternary_param_query = (
                            (where('phase_name') == param['phase_name']) & \
                            (where('parameter_type') == \
                                param['parameter_type']) & \
                            (where('constituent_array') == \
                                param['constituent_array'])
                        )
                        other_tern_params = param_search(ternary_param_query)
                        if len(other_tern_params) == 1 and \
                            other_tern_params[0] == param:
                            # only the current parameter is specified
                            # We need to generate the other two parameters.
                            order_one = copy.deepcopy(param)
                            order_one['parameter_order'] = 1
                            order_two = copy.deepcopy(param)
                            order_two['parameter_order'] = 2
                            # Add these parameters to our iteration.
                            params.extend((order_one, order_two))
                    # Include variable indicated by parameter order index
                    # Perform Muggianu adjustment to site fractions
                    mixing_term *= comp_symbols[param['parameter_order']].subs(
                        self._Muggianu_correction_dict(comp_symbols),
                        simultaneous=True)
            # Special normalization rules for parameters apply under this model
            # If there are no anions present in the anion sublattice (only VA and neutral
            # species), then the energy has an additional Q*y(VA) term
            anions_present = any([m.species.charge < 0 for m in mixing_term.free_symbols])
            if not anions_present:
                pair_rule = {}
                # Cation site fractions must always appear with vacancy site fractions
                va_subls = [(v.Species('VA') in phase.constituents[idx]) for idx in range(len(phase.constituents))]
                # The last index that contains a vacancy
                va_subl_idx = (len(phase.constituents) - 1) - va_subls[::-1].index(True)
                va_present = any((v.Species('VA') in c) for c in param['constituent_array'])
                if va_present and (max(len(c) for c in param['constituent_array']) == 1):
                    # No need to apply pair rule for VA-containing endmember
                    pass
                elif va_subl_idx > -1:
                    for sym in mixing_term.free_symbols:
                        if sym.species.charge > 0:
                            pair_rule[sym] = sym * v.SiteFraction(sym.phase_name, va_subl_idx, v.Species('VA'))
                mixing_term = mixing_term.xreplace(pair_rule)
                # This parameter is normalized differently due to the variable charge valence of vacancies
                mixing_term *= self.site_ratios[va_subl_idx]
            param_val = param['parameter']
            if isinstance(param_val, Piecewise):
                # Eliminate redundant Piecewise and extrapolate beyond temperature limits
                filtered_args = [i for i in param_val.args if not ((i.cond == S.true) and (i.expr == S.Zero))]
                if len(filtered_args) == 1:
                    param_val = filtered_args[0].expr
            rk_terms.append(mixing_term * param_val)
        return Add(*rk_terms)
