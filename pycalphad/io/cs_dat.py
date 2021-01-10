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


class TokenParser(deque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_index = 0

    def parse(self, cls: type):
        self.token_index += 1
        return cls(self.popleft())

    def parseN(self, N: int, cls: type):
        if N < 1:
            raise ValueError(f'N must be >=1, got {N}')
        return [self.parse(cls) for _ in range(N)]


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
    return TokenParser('\n'.join(instring.splitlines()[startline:]).split())


def parse_header(toks: TokenParser):
    num_pure_elements = toks.parse(int)
    num_soln_phases = toks.parse(int)
    list_soln_species_count = toks.parseN(num_soln_phases, int)
    num_stoich_phases = toks.parse(int)
    pure_elements = toks.parseN(num_pure_elements, str)
    pure_elements_mass = toks.parseN(num_pure_elements, float)
    num_gibbs_coeffs = toks.parse(int)
    gibbs_coefficient_idxs = toks.parseN(num_gibbs_coeffs, int)
    num_excess_coeffs = toks.parse(int)
    excess_coefficient_idxs = toks.parseN(num_excess_coeffs, int)
    header = Header(list_soln_species_count, num_stoich_phases, pure_elements, pure_elements_mass, gibbs_coefficient_idxs, excess_coefficient_idxs)
    return header


def parse_interval(toks: TokenParser, num_gibbs_coeffs, has_additional_terms):
    temperature = toks.parse(float)
    coefficients = toks.parseN(num_gibbs_coeffs, float)
    if has_additional_terms:
        num_additional_terms = toks.parse(int)
        additional_coeff_pairs = [AdditionalCoefficientPair(*toks.parseN(2, float)) for _ in range(num_additional_terms)]
    else:
        additional_coeff_pairs = []
    return Interval(temperature, coefficients, additional_coeff_pairs)


def parse_endmember(toks: TokenParser, num_pure_elements, num_gibbs_coeffs):
    species_name = toks.parse(str)
    gibbs_eq_type = toks.parse(int)
    has_additional_terms = gibbs_eq_type in (4,)
    num_intervals = toks.parse(int)
    stoichiometry_pure_elements = toks.parseN(num_pure_elements, float)
    intervals = [parse_interval(toks, num_gibbs_coeffs, has_additional_terms) for _ in range(num_intervals)]
    return Endmember(species_name, gibbs_eq_type, stoichiometry_pure_elements, intervals)


def parse_endmember_subq(toks: TokenParser, num_pure_elements, num_gibbs_coeffs):
    em = parse_endmember(toks, num_pure_elements, num_gibbs_coeffs)
    # TODO: is 5 correct? I only have two SUBQ/SUBG databases and they seem equivalent
    # I think the first four are the actual stoichiometries of each element in the quadruplet, but I'm unclear.
    stoichiometry_quadruplet = toks.parseN(5, float)
    coordination = toks.parse(float)
    return EndmemberSUBQ(em.species_name, em.gibbs_eq_type, em.stoichiometry_pure_elements, em.intervals, stoichiometry_quadruplet, coordination)


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
