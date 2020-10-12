"""
Support for reading ChemSage DAT files.

This implementation is based on a careful reading of Thermochimica source code (with some trial-and-error).
Thermochimica is software written by M. H. A. Piro and released under the BSD License.

"""

from pycalphad import Database
from pycalphad.io.grammar import float_number, chemical_formula
import pycalphad.variables as v
from pyparsing import CaselessKeyword, CharsNotIn, Forward, Group, Combine, Empty
from pyparsing import StringEnd, LineEnd, MatchFirst, OneOrMore, Optional, Regex, SkipTo
from pyparsing import ZeroOrMore, Suppress, White, Word, alphanums, alphas, nums
from pyparsing import delimitedList, ParseException
from functools import partial

int_number = Word(nums).setParseAction(lambda t: [int(t[0])])
species_name = Word(alphanums + '()')
phase_name = Word(alphanums + '_')
stoi_phase_name = Combine(phase_name + '(s)')
solution_phases_block = Forward()
stoichiometric_phases_block = Forward()
species_solution_integer_list = Forward()

header_species_block = Forward()


def create_gibbs_equation_block(phase_id, block, magnetic_terms, toks):
    num_additional_terms = int(toks[str(phase_id)+'_num_additional_terms'])
    eq_type = int(toks[str(phase_id) + '_gibbs_eq_type'])
    block << Group(Group(7 * float_number) +
                 num_additional_terms * (Group(CaselessKeyword('1') + (2 * float_number) + Optional(Group(7 * float_number))) |
                                         Group(CaselessKeyword('2') + (4 * float_number) + Group(7 * float_number))
                                         )
                 )
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
    for phase_idx in range(num_solution_phases):
        num_species = num_species_in_solution_phase[phase_idx]
        gibbs_equation_block = Forward()
        gibbs_magnetic_terms = Forward()
        species_block = Group(species_name + (int_number(str(phase_idx) + '_gibbs_eq_type') +\
                        int_number(str(phase_idx) + '_num_additional_terms')).\
                            addParseAction(partial(create_gibbs_equation_block, phase_idx,
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
header_preamble.addParseAction(create_solution_phase_blocks)

# TODO: Is this always just "6   1   2   3   4   5   6" twice?
header_gibbs_temperature_terms = Group(7 * int_number) + Group(7 * int_number)

header_block = header_preamble + header_species_block + header_gibbs_temperature_terms
data_block = solution_phases_block + stoichiometric_phases_block

cs_dat_grammar = header_block + data_block + SkipTo(StringEnd())


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
    try:
        result = cs_dat_grammar.parseString(lines)
        result.pprint()
    except ParseException as e:
        print(e.line)
        print(' ' * (e.col - 1) + '^')
        print(e)


def old_read_cs_dat(dbf, fd):
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
    lines = lines.replace('\t', ' ')
    lines = lines.strip()
    # Split the string by newlines
    splitlines = lines.split('\n')
    # Remove extra whitespace inside line
    splitlines = [' '.join(k.split()) for k in splitlines]

    # Header Block
    # Line 0: System title (not stored)
    current_index = 0

    # Line 1: Number of elements, Number of solution phases
    current_index = 1
    line = splitlines[current_index].split(' ')
    number_of_elements = int(line[0])
    number_of_solution_phases = int(line[1])
    gas_phase_present = int(line[2]) != 0
    if gas_phase_present:
        # Indexing must be adjusted for gaseous species
        number_of_gaseous_species = int(line[2])
        number_of_species_in_solution_phase = [number_of_gaseous_species] + [int(x) for x in line[3:3+number_of_solution_phases-1]]
        number_of_system_species = int(line[3+number_of_solution_phases-1])
    else:
        number_of_species_in_solution_phase = [int(x) for x in line[3:3+number_of_solution_phases]]
        number_of_system_species = int(line[3+number_of_solution_phases])

    # Line 2: System components
    current_index = 2
    elements = splitlines[current_index].split(' ')
    # Rename the 'electron' element to be compatible
    e_idx = -1
    for idx, element in enumerate(elements):
        if element.startswith('e('):
            e_idx = idx
            break
    if e_idx > -1:
        elements[e_idx] = '/-'
    elements = [v.Species(x) for x in elements]

    # Line 3: Component masses
    current_index = 3
    line = splitlines[current_index].split(' ')
    element_masses = [float(x) for x in line]
    element_mass_dict = dict(zip(elements, element_masses))

    # Line 4: Temperature dependence term definition
    current_index = 4
    line = [int(x) for x in splitlines[current_index].split(' ')]
    if line != [6, 1, 2, 3, 4, 5, 6]:
        raise ValueError('Unknown temperature dependence term specified in database')

    # Line 5: Temperature dependence term definition (Again? Why?)
    current_index = 5

    # Data Block
    # Line 6: Solution phases
    current_index = 6
    phase_dict = {}
    for solution_phase_idx in range(number_of_solution_phases):
        phase_name = splitlines[current_index + solution_phase_idx]
        phase_model_type = splitlines[current_index + solution_phase_idx + 1]
        if phase_model_type != 'IDMX':
            phase_model_data = splitlines[current_index + solution_phase_idx + 2]
        else:
            phase_model_data = None
        if phase_model_data is None:
            leading_index = current_index + solution_phase_idx + 2
            print(leading_index)
        else:
            leading_index = current_index + solution_phase_idx + 3
            print(leading_index)
        num_species = number_of_species_in_solution_phase[solution_phase_idx]
        species_dict = {}
        for species_idx in range(num_species):
            species_name = splitlines[leading_index]
            line = splitlines[leading_index + 1].split(' ')
            equation_type = int(line[0])
            num_additional_eq_entries = int(line[1])
            species_elemental_stoichiometry = [float(x) for x in line[2:]]
            # TODO: Remove the zero-constituents from the constituents dict
            species_entry = v.Species(species_name, constituents=dict(zip(elements, species_elemental_stoichiometry)))
            line = splitlines[leading_index + 2].split(' ')
            # There should be exactly 7 entries here
            gibbs_coefficients = [float(x) for x in line]
            if len(gibbs_coefficients) < 7:
                line = splitlines[leading_index + 3].split(' ')
                gibbs_coefficients.extend([float(x) for x in line])
                leading_index += 1
                print(leading_index)
            leading_index = leading_index + 3
            print(leading_index)
            for eq_species_idx in range(num_additional_eq_entries):
                line = splitlines[leading_index].split(' ')
                coef_header = [float(x) for x in line]
                print(coef_header)
                if coef_header == [1.0, 0., 0.]:
                    leading_index += 1
                    print(leading_index)
                    break
                second_line = splitlines[leading_index + 1].split(' ')
                second_line = [float(x) for x in second_line]
                if len(second_line) < 7:
                    line = splitlines[leading_index + 2].split(' ')
                    second_line.extend([float(x) for x in line])
                    leading_index += 1
                    print(leading_index)
                gibbs_coefficients.append([coef_header, second_line])
                leading_index += 1
                print(leading_index)
            species_dict[species_entry] = {'equation_type': equation_type, 'gibbs_coefficients': gibbs_coefficients}
        phase_dict[phase_name] = {'phase_model_type': phase_model_type, 'phase_model_data': phase_model_data,
                                  'species': species_dict}


Database.register_format("dat", read=read_cs_dat, write=None)
