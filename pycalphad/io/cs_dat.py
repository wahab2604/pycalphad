"""
Support for reading ChemSage DAT files.

This implementation is based on a careful reading of Thermochimica source code (with some trial-and-error).
Thermochimica is software written by M. H. A. Piro and released under the BSD License.

Careful of a gotcha:  `obj.setParseAction` modifies the object in place but calling it a name makes a new object:

`obj = Word(nums); obj is obj.setParseAction(lambda t: t)` is True
`obj = Word(nums); obj is obj('abc').setParseAction(lambda t: t)` is False



"""


from pycalphad import Database
from pycalphad.io.grammar import float_number, chemical_formula
import pycalphad.variables as v
from pyparsing import CaselessKeyword, CharsNotIn, Forward, Group, Combine, Empty
from pyparsing import StringEnd, LineEnd, MatchFirst, OneOrMore, Optional, Regex, SkipTo
from pyparsing import ZeroOrMore, Suppress, White, Word, alphanums, alphas, nums
from pyparsing import delimitedList, ParseException

int_number = Word(nums).setParseAction(lambda t: [int(t[0])])
species_name = Word(alphanums + '()')
phase_name = Word(alphanums + '_')
stoi_phase_name = Word(alphanums + '_()')


def parseN(parser_element):
    """Return a new parser element that parses N and N instances of parser_element N (N*parser_element)

    Examples
    --------
    >>> from pycalphad.io.cs_dat import int_number, parseN, Word, alphas
    >>> out = parseN(int_number).parseString("6 1 2 3 4 5 6")
    >>> assert out.asList() == [1, 2, 3, 4, 5, 6]
    >>> out = parseN(Word(alphas)).parseString("2 hello world")
    >>> assert out.asList() == ["hello", "world"]

    """
    exprs = Forward()

    def _setup_N_exprs(toks):
        _N = int(toks[0])
        exprs << _N*parser_element

    N = int_number().setParseAction(_setup_N_exprs)
    return Suppress(N) + exprs


def grammar_header():
    """Define the grammar to parse the header"""

    # some forward definitions
    fwd_list_soln_species_count = Forward()
    def _set_list_soln_species_count(self, toks):
        """Set fwd_list_soln_species_count depending on how many phases there are"""
        toks['num_soln_phases'] = int(toks['num_soln_phases'])
        fwd_list_soln_species_count << Group(toks['num_soln_phases'] * int_number)('list_soln_species_count')
        return [toks['num_soln_phases']]

    fwd_species_list = Forward()
    fwd_species_masses = Forward()
    def _set_species_list_masses(self, toks):
        num_elements = int(toks['num_elements'])
        # Element names, followed by their masses
        fwd_species_list << Group(num_elements * species_name)
        fwd_species_masses << Group(num_elements * float_number)

    # comment line
    comment_line = Suppress(SkipTo(LineEnd()))

    # phases line
    num_elements = int_number('num_elements').setParseAction(_set_species_list_masses)
    num_soln_phases = int_number('num_soln_phases').setParseAction(_set_list_soln_species_count)
    num_stoich_phases = int_number('num_stoich_phases')
    phases_line = Suppress(num_elements) + Suppress(num_soln_phases) + fwd_list_soln_species_count + num_stoich_phases

    # species line (really a "block" over multiple lines)
    species_line = fwd_species_list('pure_elements')

    # species mass line (really a "block" over multiple lines)
    species_mass_line = fwd_species_masses('pure_elements_mass')

    # gibbs coeffiecients indices line
    gibbs_line = Group(parseN(int_number()))('gibbs_coefficient_idxs')

    # excess coeffiecients indices line
    excess_line = Group(parseN(int_number()))('excess_coefficient_idxs')

    # combined header
    header = (
        comment_line +
        phases_line +
        species_line +
        species_mass_line +
        gibbs_line +
        excess_line
    )
    return header


def parse_header(string):
    pass


def parse_cs_dat(lines):
    # Procedurally parse chemsage files
    pass




class ChemsageGrammar():
    """
    By convention:

    Methods with a single leading underscore construct grammars
    Methods with two leading underscores are intended to return functions which may be used as parse actions (which may set internal state)
    """

    def __init__(self):
        # Forward definitions to be used later
        self.fwd_solution_phases_block = Forward()
        self.fwd_stoichiometric_phases_block = Forward()
        self.fwd_species_solution_integer_list = Forward()
        self.fwd_header_species_block = Forward()

    def __setup_blocks(self, toks):
        num_elements = toks['number_elements']
        # Element names, followed by their masses
        self.fwd_header_species_block << (num_elements * species_name) + (num_elements * float_number)
        self.fwd_header_species_block.addParseAction(lambda t: [dict(zip(t[:num_elements], t[num_elements:]))])

    def __set_species_array_size(self, toks):
        toks['number_solution_phases'] = int(toks['number_solution_phases'])
        self.fwd_species_solution_integer_list << Group(toks['number_solution_phases'] * int_number)('number_species_in_solution_phase')
        return [toks['number_solution_phases']]

    def _coefficients_parse_block(self, name):
        """Return a parse for reading a string of the form:

        `N [N-integers]...`

        where the first number describes how many integer coefficients follow. Examples are:

        ```
        1 1
        2 1 2
        3 1 2 3
        6 1 2 3 4 5 6
        4 3 4 5 6
        ```
        """
        fwd_coefficients = Forward()

        def setup_coefficients(num_coeffs_toks):
            # gets the first token and extracts the integer value from the ParseResults
            num_coefficients = int(num_coeffs_toks[0][0])
            fwd_coefficients << Group(num_coefficients*int_number)
            return num_coeffs_toks
        return Suppress(int_number().setParseAction(setup_coefficients)) + fwd_coefficients(name)

    @staticmethod
    def _excess_block(num_excess_coeffs):
        """
        Create a block to parse interaction parameters

        Parses the following:
        ```
         2
         1   2   3
        30965.780     -27.614000     0.00000000     0.00000000
        11865.820      8.2801000     0.00000000     0.00000000
        20145.960     0.00000000     0.00000000     0.00000000
         2
         1   3   2
        -38417.500     0.00000000     0.46870000E-01 -.19131800E-04
        -1472.7700     0.00000000     -.31652000E-02 0.25690000E-05
        0
        ```

        which can be accessed via
        ```
        out = pg.parseString(excess_params_str)
        print(out['excess_terms'][0]['interacting_species_ids'])
        print(out['excess_terms'][0]['num_interaction_parameters'])
        print(out['excess_terms'][0]['interaction_parameters'])
        ```

        ```
        [1, 2]
        3
        [[30965.78, -27.614, 0.0, 0.0], [11865.82, 8.2801, 0.0, 0.0], [20145.96, 0.0, 0.0, 0.0]]

        The number of cofficients in each excess parameter is determined by the header and must be passed in here.
        ```
        """
        interacting_species_ids = Forward()
        interaction_parameters = Forward()

        def create_interacting_species_ids(toks):
            num_interacting_species = int(toks['num_interacting_species'])
            if num_interacting_species != 0:
                # need to special case, because 0 is the terminator
                interacting_species_ids << Group(num_interacting_species*int_number)('interacting_species_ids')
            return toks

        def create_interaction_parameters(toks):
            interaction_parameter_coefficients = Group(num_excess_coeffs*float_number)
            num_interaction_parameters = int(toks['num_interaction_parameters'])
            interaction_parameters << Group(num_interaction_parameters*interaction_parameter_coefficients)('interaction_parameters')
            return toks

        num_interacting_species = int_number('num_interacting_species').setParseAction(create_interacting_species_ids)
        num_interaction_parameters = int_number('num_interaction_parameters').setParseAction(create_interaction_parameters)

        excess_term = num_interacting_species + \
            interacting_species_ids + num_interaction_parameters + \
            interaction_parameters
        return OneOrMore(Suppress('0') | Group(excess_term))('excess_terms')

    @staticmethod
    def __create_gibbs_equation_block(phase_id, num_gibbs_coeffs, num_excess_coeffs, fwd_block, magnetic_terms):
        def f(toks):
            num_additional_terms = int(toks[str(phase_id)+'_num_additional_terms'])
            eq_type = int(toks[str(phase_id) + '_gibbs_eq_type'])
            if eq_type not in (1, 4, 13, 16):
                raise ValueError(f"Only Gibbs energy equation types of 1, 4, 13 or 16 are supported. Found {eq_type} for phase id {phase_id}")

            if (eq_type % 12) in (4, 5, 6, 10, 11, 12):
                # additional terms after each line where an integer number of (coefficient, exponent) pairs added

                # create the block for the additional (coefficient, exponent) pairs
                # the number of pairs that are present is indicated by the first integer number in the block
                coeff_exponent_pairs = Forward()

                def set_pairs(num_pairs_toks):
                    # gets the first token and extracts the integer value from the ParseResults
                    number_of_pairs = int(num_pairs_toks[0][0])
                    coeff_exponent_pairs << Group(number_of_pairs*Group(float_number('coefficient') + float_number('exponent')))('coefficient_exponent_pairs')
                    return num_pairs_toks
                additional_terms_block = Suppress(int_number('num_coeff_exp_pairs').setParseAction(set_pairs)) + coeff_exponent_pairs
                # parse the Gibbs coefficients plus the temperature upper limit
                fwd_block << num_additional_terms * Group(float_number('temperature_limit') + Group(num_gibbs_coeffs * float_number)('gibbs_coefficients') + additional_terms_block('additional_terms'))
            else:
                # parse the Gibbs coefficients plus the temperature upper limit
                fwd_block << num_additional_terms * Group(float_number('temperature_limit') + Group(num_gibbs_coeffs * float_number)('gibbs_coefficients'))
            if eq_type == 16:
                if phase_id == 'temp':
                    # stoichiometric phase
                    magnetic_terms << Group(4 * float_number)(str(phase_id)+'magnetic_terms')
                else:
                    # solution phase
                    magnetic_terms << Group(2 * float_number)(str(phase_id) + 'magnetic_terms')
            else:
                magnetic_terms << Empty()
            return [toks.species_name, eq_type, num_additional_terms]
        return f

    @staticmethod
    def _solution_phase_header_block():
        # Special cases for different models
        SUBG_data = float_number('FFN_SNN_ratio') + int_number('num_pairs') + int_number('num_non_default_quadruplets')
        SUBQ_data = int_number('num_pairs') + int_number('num_non_default_quadruplets')
        phase_header = phase_name + Word(alphanums)('model_name') + Optional(SUBG_data | SUBQ_data)
        return phase_header

    @staticmethod
    def _species_reference_energy_block(phase_idx, num_gibbs_coeffs, num_excess_coeffs, num_elements):
        """Returns a grammar that parses the reference energy for a single species"""
        phase_idx_str = str(phase_idx)

        # Define the header, i.e.
        # CuCl
        #   1  1    0.0    0.0    1.0    0.0    1.0    0.0    0.0    0.0
        phase_gibbs_eq_type = int_number(phase_idx_str + '_gibbs_eq_type')
        phase_num_addit_terms = int_number(phase_idx_str + '_num_additional_terms')
        pair_stoichiometry = Group(num_elements * float_number)('pair_stoichiometry')
        species_header = species_name('species_name') + phase_gibbs_eq_type + phase_num_addit_terms + pair_stoichiometry

        # We use the header to set up the Gibbs energy blocks via the forwarded definitions, i.e. the terms like
        #  1000.0000     -5219.3324     -12.179856     -26.924057     -.84893408E-02
        # 0.11276942E-05 -114664.63
        # 1 -316.64663       0.50
        gibbs_equation_block = Forward()
        gibbs_magnetic_terms = Forward()
        gibbs_eq_block_func = ChemsageGrammar.__create_gibbs_equation_block(phase_idx, num_gibbs_coeffs, num_excess_coeffs, gibbs_equation_block, gibbs_magnetic_terms)
        species_header.addParseAction(gibbs_eq_block_func)

        # Finally, we construct the entire block
        species_block = Group(species_header + gibbs_equation_block + gibbs_magnetic_terms)
        return species_block

    def _SUBQ_species_reference_energy_block(phase_idx, num_gibbs_coeffs, num_excess_coeffs, num_elements):
        """Returns a grammar that parses the reference energy for a single species"""
        phase_idx_str = str(phase_idx)

        # Define the header, i.e.
        # CuCl
        #   1  1    0.0    0.0    1.0    0.0    1.0    0.0    0.0    0.0
        phase_gibbs_eq_type = int_number(phase_idx_str + '_gibbs_eq_type')
        phase_num_addit_terms = int_number(phase_idx_str + '_num_additional_terms')
        pair_stoichiometry = Group(num_elements * float_number)('pair_stoichiometry')
        species_header = species_name('species_name') + phase_gibbs_eq_type + phase_num_addit_terms + pair_stoichiometry

        # We use the header to set up the Gibbs energy blocks via the forwarded definitions, i.e. the terms like
        #  1000.0000     -5219.3324     -12.179856     -26.924057     -.84893408E-02
        # 0.11276942E-05 -114664.63
        # 1 -316.64663       0.50
        gibbs_equation_block = Forward()
        gibbs_magnetic_terms = Forward()
        gibbs_eq_block_func = ChemsageGrammar.__create_gibbs_equation_block(phase_idx, num_gibbs_coeffs, num_excess_coeffs, gibbs_equation_block, gibbs_magnetic_terms)
        species_header.addParseAction(gibbs_eq_block_func)

        # SUBQ has an extra term for the stoichiometry of the pair in terms of the constituents of the pair
        # they seem to always come in groups of 5
        pair_constituent_stoichiometry = Group(5*float_number)('pair_constituent_stoichiometry')
        coordination_number = Optional(float_number('coordination_number'))  # Optional for SUBG, which specifies it with the phase

        # Finally, we construct the entire block
        species_block = Group(species_header + gibbs_equation_block + gibbs_magnetic_terms + pair_constituent_stoichiometry + coordination_number)
        return species_block

    @staticmethod
    def _solution_phase_block(num_species_in_phase: int, num_excess_coeffs: int,
                              # species block only
                              phase_idx: int, num_gibbs_coeffs: int, num_elements: int,
                              ):
        header = ChemsageGrammar._solution_phase_header_block()
        species_block = ChemsageGrammar._species_reference_energy_block(phase_idx, num_gibbs_coeffs, num_excess_coeffs, num_elements)
        return Group(header + Group(num_species_in_phase * species_block)('surface_reference_terms') + Optional(ChemsageGrammar._excess_block(num_excess_coeffs))('excess_terms'))

    @staticmethod
    def _SUBQ_solution_phase_block(num_species_in_phase: int, num_excess_coeffs: int,
                                   # species block only
                                   phase_idx: int, num_gibbs_coeffs: int, num_elements: int,
                                   ):
        header = ChemsageGrammar._solution_phase_header_block()
        species_block = ChemsageGrammar._SUBQ_species_reference_energy_block(phase_idx, num_gibbs_coeffs, num_excess_coeffs, num_elements)
        return Group(header + Group(num_species_in_phase * species_block)('surface_reference_terms') + Optional(ChemsageGrammar._excess_block(num_excess_coeffs))('excess_terms'))

    def __create_solution_phase_blocks(self, toks):
        num_elements = toks['number_elements']
        num_solution_phases = toks['number_solution_phases']
        num_species_in_solution_phase = toks['number_species_in_solution_phase']
        if num_species_in_solution_phase[0] == 0:
            # gas phase is not included and the zero is a placeholder.
            num_solution_phases = num_solution_phases - 1
            num_species_in_solution_phase = num_species_in_solution_phase[1:]
        num_gibbs_coeffs = len(toks['gibbs_coefficient_idxs'])
        num_excess_coeffs = len(toks['excess_coefficient_idxs'])
        soln_phase_blocks = []
        for phase_idx in range(num_solution_phases):
            num_species = num_species_in_solution_phase[phase_idx]
            soln_phase_block = self._solution_phase_block(num_species, num_excess_coeffs, phase_idx, num_gibbs_coeffs, num_elements)
            SUBQ_soln_phase_block = self._SUBQ_solution_phase_block(num_species, num_excess_coeffs, phase_idx, num_gibbs_coeffs, num_elements)
            soln_phase_blocks.append(soln_phase_block | SUBQ_soln_phase_block)
        soln_phase_expr = Empty()
        for soln_phase_block in soln_phase_blocks:
            soln_phase_expr = soln_phase_expr + soln_phase_block
        self.fwd_solution_phases_block << soln_phase_expr
        stoi_gibbs_equation_block = Forward()
        stoi_magnetic_terms = Forward()
        stoi_gibbs_block = Group((int_number('temp_gibbs_eq_type') + int_number('temp_num_additional_terms')).addParseAction(self.__create_gibbs_equation_block('temp', num_gibbs_coeffs, num_excess_coeffs, stoi_gibbs_equation_block, stoi_magnetic_terms)) + Group(num_elements * float_number) + stoi_gibbs_equation_block + stoi_magnetic_terms)
        self.fwd_stoichiometric_phases_block << ZeroOrMore(Group(stoi_phase_name + Optional('#') + Group(stoi_gibbs_block)))

    def _header_block(self):
        comment_line = Suppress(SkipTo(LineEnd()))
        num_elements = int_number('number_elements')
        number_solution_phases = int_number('number_solution_phases').setParseAction(self.__set_species_array_size)
        number_species_in_system = int_number('number_species_in_system')
        header_preamble = comment_line + num_elements + number_solution_phases + self.fwd_species_solution_integer_list + number_species_in_system
        header_preamble.addParseAction(self.__setup_blocks)

        header_block = header_preamble + \
            self.fwd_header_species_block + \
            self._coefficients_parse_block('gibbs_coefficient_idxs') + \
            self._coefficients_parse_block('excess_coefficient_idxs')
        header_block.addParseAction(self.__create_solution_phase_blocks)
        return header_block

    def _phases_block(self):
        """The block containing phase definitions"""
        solution_phases_block = Group(self.fwd_solution_phases_block)('solution_phases')
        stoichiometric_phases_block = Group(self.fwd_stoichiometric_phases_block)('stoichiometric_phases')
        return solution_phases_block + stoichiometric_phases_block

    def get_grammar(self):
        """Return the pyparsing grammar for Chemsage-style .dat files"""
        # TODO: Currently skips to end to be able to support comments at end of
        # files. This can lead to the grammar failing to parse a certain
        # block and treating the rest of the file as a comment.
        grammar = self._header_block() + self._phases_block() + SkipTo(StringEnd())
        return grammar



def create_cs_dat_grammar():
    return ChemsageGrammar().get_grammar()

def read_cs_dat(dbf, fd):
    """
    Parse a ChemSage DAT file into a pycalphad Database object.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    """
    lines = fd.read()
    cs_dat_grammar = ChemsageGrammar().get_grammar()
    try:
        result = cs_dat_grammar.parseString(lines)
        # result.pprint()
        # print('pass ', end='')
    except ParseException as e:
        raise e
        # print('fail ', end='')
        # print(e.line)
        # print(' ' * (e.col - 1) + '^')
        # print(e)
    except Exception as e:
        # print('fail ', end='')
        raise e


Database.register_format("dat", read=read_cs_dat, write=None)