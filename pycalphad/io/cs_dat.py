"""
Support for reading ChemSage DAT files.

This implementation is based on a careful reading of Thermochimica source code (with some trial-and-error).
Thermochimica is software written by M. H. A. Piro and released under the BSD License.

Careful of a gotcha:  `obj.setParseAction` modifies the object in place but calling it a name makes a new object:

`obj = Word(nums); obj is obj.setParseAction(lambda t: t)` is True
`obj = Word(nums); obj is obj('abc').setParseAction(lambda t: t)` is False



"""

from dataclasses import dataclass
from pycalphad import Database
from collections import namedtuple, deque


def popleftN(_deque, N):
    if N < 1:
        raise ValueError('N must be >=1')
    return (_deque.popleft() for _ in range(N))


def parseN(_deque, N, cls):
    if N < 1:
        raise ValueError('N must be >=1')
    return [cls(x) for x in popleftN(_deque, N)]


def parse(_deque, cls):
    return cls(_deque.popleft())


Header = namedtuple('Header', ('list_soln_species_count', 'num_stoich_phases', 'pure_elements', 'pure_elements_mass', 'gibbs_coefficient_idxs', 'excess_coefficient_idxs'))
AdditionalCoefficientPair = namedtuple('AdditionalCoefficientPair', ('coefficient', 'exponent'))
Interval = namedtuple('Interval', ('temperature', 'coefficients', 'additional_coeff_pairs'))


@dataclass
class Endmember:
    species_name: str
    gibbs_eq_type: str
    stoichiometry_pure_elements: [float]
    intervals: [Interval]


@dataclass
class EndmemberSUBQ:
    stoichiometry_pairs: [float]
    coordination: float


@dataclass
class _Phase:
    phase_name: str
    phase_type: str
    endmembers: [Endmember]


@dataclass
class Phase_SUBQ(_Phase):
    num_pairs: int
    num_quadruplets: int
    num_subl_1_consts: int
    num_subl_2_const: int


def tokenize(instring, startline=0):
    return deque('\n'.join(instring.splitlines()[startline:]).split())


def parse_header(toks: deque):
    num_pure_elements = parse(toks, int)
    num_soln_phases = parse(toks, int)
    list_soln_species_count = parseN(toks, num_soln_phases, int)
    num_stoich_phases = parse(toks, int)
    pure_elements = parseN(toks, num_pure_elements, str)
    pure_elements_mass = parseN(toks, num_pure_elements, float)
    num_gibbs_coeffs = parse(toks, int)
    gibbs_coefficient_idxs = parseN(toks, num_gibbs_coeffs, int)
    num_excess_coeffs = parse(toks, int)
    excess_coefficient_idxs = parseN(toks, num_excess_coeffs, int)
    header = Header(list_soln_species_count, num_stoich_phases, pure_elements, pure_elements_mass, gibbs_coefficient_idxs, excess_coefficient_idxs)
    return header


def parse_interval(toks: deque, num_gibbs_coeffs, has_additional_terms):
    temperature = parse(toks, float)
    coefficients = parseN(toks, num_gibbs_coeffs, float)
    if has_additional_terms:
        num_additional_terms = parse(toks, int)
        additional_coeff_pairs = [AdditionalCoefficientPair(*parseN(toks, 2, float)) for _ in range(num_additional_terms)]
    else:
        additional_coeff_pairs = []
    return Interval(temperature, coefficients, additional_coeff_pairs)


def parse_endmember(toks: deque, num_pure_elements, num_gibbs_coeffs):
    species_name = parse(toks, str)
    gibbs_eq_type = parse(toks, int)
    has_additional_terms = gibbs_eq_type in (4,)
    num_intervals = parse(toks, int)
    stoichiometry_pure_elements = parseN(toks, num_pure_elements, float)
    intervals = [parse_interval(toks, num_gibbs_coeffs, has_additional_terms) for _ in range(num_intervals)]
    return Endmember(species_name, gibbs_eq_type, stoichiometry_pure_elements, intervals)


def parse_cs_dat(instring):
    toks = tokenize(instring, startline=1)
    header = parse_header(toks)
    solution_phases = []
    stoichiometric_phases = []
    # TODO: better handling and validation of this
    remaining_tokens = ' '.join(toks)
    print(remaining_tokens)
    return header, solution_phases, stoichiometric_phases


def read_cs_dat(dbf: Database, fd):
    """
    Parse a ChemSage DAT file into a pycalphad Database object.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    """
    data = parse_cs_dat(fd.read())
    # TODO: modify the dbf in place


Database.register_format("dat", read=read_cs_dat, write=None)