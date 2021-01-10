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
class Quadruplet:
    quadruplet_idxs: [int]  # exactly four
    quadruplet_coordinations: [float]  # exactly four


@dataclass
class ExcessQuadruplet:
    mixing_type: int
    mixing_code: str
    mixing_const: [int]  # exactly four
    mixing_exponents: [int]  # exactly four
    junk: [float]  # exactly twelve
    additional_mixing_const: int
    additional_mixing_exponent: int
    excess_coeffs: [float]


@dataclass
class Endmember:
    species_name: str
    gibbs_eq_type: str
    stoichiometry_pure_elements: [float]
    intervals: [Interval]


@dataclass
class EndmemberSUBQ(Endmember):
    stoichiometry_quadruplet: [float]
    coordination: float


@dataclass
class ExcessCEF:
    dummy: str


@dataclass
class _Phase:
    phase_name: str
    phase_type: str
    endmembers: [Endmember]


@dataclass
class Phase_Stoichiometric:
    phase_name: str
    phase_type: str
    endmembers: [Endmember]  # exactly one


@dataclass
class Phase_CEF(_Phase):
    excess_parameters: [ExcessCEF]


@dataclass
class Phase_SUBQ(_Phase):
    num_pairs: int
    num_quadruplets: int
    num_subl_1_const: int
    num_subl_2_const: int
    subl_1_const: [str]
    subl_2_const: [str]
    subl_1_charges: [float]
    subl_1_chemical_groups: [int]
    subl_2_charges: [float]
    subl_2_chemical_groups: [int]
    subl_const_idx_pairs: [(int,)]
    quadruplets: [Quadruplet]
    excess_parameters: [ExcessQuadruplet]


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
    has_magnetic = gibbs_eq_type > 12
    has_additional_terms = ((gibbs_eq_type - 12) if has_magnetic else gibbs_eq_type) in (4,)
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


def parse_quadruplet(toks):
    quad_idx = toks.parseN(4, int)
    quad_coords = toks.parseN(4, float)
    return Quadruplet(quad_idx, quad_coords)


def parse_subq_excess(toks, mixing_type, num_excess_coeffs):
    mixing_code = toks.parse(str)
    mixing_const = toks.parseN(4, int)
    mixing_exponents = toks.parseN(4, int)
    junk = toks.parseN(12, float)
    additional_mixing_const = toks.parse(int)
    additional_mixing_exponent = toks.parse(float)
    excess_coeffs = toks.parseN(num_excess_coeffs, float)
    return ExcessQuadruplet(mixing_type, mixing_code, mixing_const, mixing_exponents, junk, additional_mixing_const, additional_mixing_exponent, excess_coeffs)


def parse_phase_subq(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs):
    num_pairs = toks.parse(int)
    num_quadruplets = toks.parse(int)
    endmembers = [parse_endmember_subq(toks, num_pure_elements, num_gibbs_coeffs) for _ in range(num_pairs)]
    num_subl_1_const = toks.parse(int)
    num_subl_2_const = toks.parse(int)
    subl_1_const = toks.parseN(num_subl_1_const, str)
    subl_2_const = toks.parseN(num_subl_2_const, str)
    subl_1_charges = toks.parseN(num_subl_1_const, float)
    subl_1_chemical_groups = toks.parseN(num_subl_1_const, int)
    subl_2_charges = toks.parseN(num_subl_2_const, float)
    subl_2_chemical_groups = toks.parseN(num_subl_2_const, int)
    # TODO: not exactly sure my math is right on how many pairs, but I think it should be cations*anions
    subl_1_pair_idx = toks.parseN(num_subl_1_const*num_subl_2_const, int)
    subl_2_pair_idx = toks.parseN(num_subl_1_const*num_subl_2_const, int)
    subl_const_idx_pairs = [(s1i, s2i) for s1i, s2i in zip(subl_1_pair_idx, subl_2_pair_idx)]
    quadruplets = [parse_quadruplet(toks) for _ in range(num_quadruplets)]
    excess_parameters = []
    while True:
        mixing_type = toks.parse(int)
        if mixing_type == 0:
            break
        excess_parameters.append(parse_subq_excess(toks, mixing_type, num_excess_coeffs))

    return Phase_SUBQ(phase_name, phase_type, endmembers, num_pairs, num_quadruplets, num_subl_1_const, num_subl_2_const, subl_1_const, subl_2_const, subl_1_charges, subl_1_chemical_groups, subl_2_charges, subl_2_chemical_groups, subl_const_idx_pairs, quadruplets, excess_parameters)


def parse_phase_cef(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs, num_const):
    endmembers = [parse_endmember(toks, num_pure_elements, num_gibbs_coeffs) for _ in range(num_const)]
    excess_parameters = []
    return Phase_CEF(phase_name, phase_type, endmembers, excess_parameters)


def parse_phase(toks, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs, num_const):
    """Dispatches to the correct parser depending on the phase type"""
    phase_name = toks.parse(str)
    phase_type = toks.parse(str)
    if phase_type == 'SUBQ':
        phase = parse_phase_subq(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs)
    elif phase_type == 'SUBG':
        raise NotImplementedError("SUBG not yet supported")
    if phase_type in ('IDMX', 'SUBL', 'RKMP'):
        # all these phases parse the same
        phase = parse_phase_cef(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs, num_const)
    return phase


def parse_stoich_phase(toks, num_pure_elements, num_gibbs_coeffs):
    endmember = parse_endmember(toks, num_pure_elements, num_gibbs_coeffs)
    phase_name = endmember.species_name
    return Phase_Stoichiometric(phase_name, None, [endmember])


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
