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
from functools import partial

def create_cs_dat_grammar():
    int_number = Word(nums).setParseAction(lambda t: [int(t[0])])
    species_name = Word(alphanums + '()')
    phase_name = Word(alphanums + '_')
    stoi_phase_name = Combine(phase_name + '(s)')
    solution_phases_block = Forward()
    stoichiometric_phases_block = Forward()
    species_solution_integer_list = Forward()

    header_species_block = Forward()

    def coefficients_parse_block(name):
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
        coefficients = Forward()
        def setup_coefficients(num_coeffs_toks):
            # gets the first token and extracts the integer value from the ParseResults
            num_coefficients = int(num_coeffs_toks[0][0])
            coefficients << Group(num_coefficients*int_number)
            return num_coeffs_toks
        return Suppress(int_number().setParseAction(setup_coefficients)) + coefficients(name)

    def create_gibbs_equation_block(phase_id, num_gibbs_coeffs, num_excess_coeffs, block, magnetic_terms, toks):
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
                coeff_exponent_pairs << Group(number_of_pairs*Group(2*float_number))
                return num_pairs_toks
            additional_terms_block = Suppress(int_number('num_coeff_exp_pairs').setParseAction(set_pairs)) + coeff_exponent_pairs
            # parse the Gibbs coefficients plus the temperature upper limit
            block << num_additional_terms * Group(Group((1 + num_gibbs_coeffs) * float_number) + additional_terms_block)
        else:
            # parse the Gibbs coefficients plus the temperature upper limit
            block << num_additional_terms * Group(Group((1 + num_gibbs_coeffs) * float_number))
        if eq_type == 16:
            if phase_id == 'temp':
                # stoichiometric phase
                magnetic_terms << Group(4 * float_number)(str(phase_id)+'magnetic_terms')
            else:
                # solution phase
                magnetic_terms << Group(2 * float_number)(str(phase_id) + 'magnetic_terms')
        else:
            magnetic_terms << Empty()
        return [eq_type, num_additional_terms]


    def create_solution_phase_blocks(toks):
        num_elements = toks['number_elements']
        num_solution_phases = toks['number_solution_phases']
        num_species_in_solution_phase = toks['number_species_in_solution_phase']
        if num_species_in_solution_phase[0] == 0:
            # gas phase is not included and the zero is a placeholder.
            num_solution_phases = num_solution_phases - 1
            num_species_in_solution_phase = num_species_in_solution_phase[1:]
        num_gibbs_coeffs = len(toks['gibbs_coefficient_idxs'])
        num_excess_coeffs = len(toks['excess_coefficient_idxs'])
        for phase_idx in range(num_solution_phases):
            num_species = num_species_in_solution_phase[phase_idx]
            gibbs_equation_block = Forward()
            gibbs_magnetic_terms = Forward()
            species_block = Group(species_name + (int_number(str(phase_idx) + '_gibbs_eq_type') +\
                            int_number(str(phase_idx) + '_num_additional_terms')).\
                                addParseAction(partial(create_gibbs_equation_block, phase_idx,
                                                       num_gibbs_coeffs, num_excess_coeffs,
                                                       gibbs_equation_block, gibbs_magnetic_terms)) +\
                            Group(num_elements * float_number) +\
                            gibbs_equation_block + gibbs_magnetic_terms)
            phase_block = Group(phase_name + Word(alphanums) + Group(num_species * species_block))
            solution_phases_block << phase_block
        stoi_gibbs_equation_block = Forward()
        stoi_magnetic_terms = Forward()
        stoi_gibbs_block = Group((int_number('temp_gibbs_eq_type') +\
                                 int_number('temp_num_additional_terms')).\
                                 addParseAction(partial(create_gibbs_equation_block, 'temp',
                                                        num_gibbs_coeffs, num_excess_coeffs,
                                                        stoi_gibbs_equation_block, stoi_magnetic_terms)) +\
                                 Group(num_elements * float_number) + \
                                 stoi_gibbs_equation_block + stoi_magnetic_terms)
        stoichiometric_phases_block << ZeroOrMore(Group(stoi_phase_name + Optional('#') + Group(stoi_gibbs_block)))


    def setup_blocks(toks):
        num_elements = toks['number_elements']
        # Element names, followed by their masses
        header_species_block << (num_elements * species_name) + (num_elements * float_number)
        header_species_block.addParseAction(lambda t: [dict(zip(t[:num_elements], t[num_elements:]))])


    def set_species_array_size(toks):
        toks['number_solution_phases'] = int(toks['number_solution_phases'])
        species_solution_integer_list << Group(toks['number_solution_phases'] * int_number)('number_species_in_solution_phase')
        return [toks['number_solution_phases']]


    header_preamble = Suppress(SkipTo(LineEnd())) + \
        int_number('number_elements') + int_number('number_solution_phases').setParseAction(set_species_array_size) + \
        species_solution_integer_list + int_number('number_species_in_system')
    header_preamble.addParseAction(setup_blocks)

    header_block = header_preamble + \
                   header_species_block + \
                   coefficients_parse_block('gibbs_coefficient_idxs') + \
                   coefficients_parse_block('excess_coefficient_idxs')
    header_block.addParseAction(create_solution_phase_blocks)
    data_block = solution_phases_block + stoichiometric_phases_block

    cs_dat_grammar = header_block + data_block + SkipTo(StringEnd())
    return cs_dat_grammar

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
    cs_dat_grammar = create_cs_dat_grammar()
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
