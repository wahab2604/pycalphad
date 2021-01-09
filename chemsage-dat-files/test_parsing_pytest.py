# coding: utf-8
import pytest

import numpy as np
import pyparsing
pyparsing.ParserElement.enablePackrat(0)
from pycalphad.io.cs_dat import create_cs_dat_grammar, grammar_header, grammar_endmember

# The number of solution phases here EXCLUDES gas phase if it's not present
# (i.e. the num_soln_phases here may be one less than line 2)
database_filenames = [
    # Reorganized to make passing ones test first
    ("Pb-Sn.dat", 4, 0),  # FACT
    ("C-O.dat", 1, 4),  # Thermochimica
    ("W-Au-Ar-Ne-O_04.dat", 2, 5),  # Thermochimica

    # Highest priority to pass:
    ("CuZnFeCl-Viitala (1).dat", 4, 11),  # https://doi.org/10.1016/j.calphad.2019.101667

    # Data files from FACT documentation
    # See https://gtt-technologies.de/software/chemapp/documentation/online-manual/
    (pytest.param("C-N-O.dat", 1, 2, marks=pytest.mark.xfail)),
    ("C-O-Si.dat", 1, 7),
    ("Fe-C.dat", 4, 2),
    ("Fe2SiO4-Mg2SiO4.dat", 3, 2),
    ("O-H-EA.dat", 2, 1),
    ("Pitzer.dat", 2, 6),
    ("subl-ex.dat", 4, 0),

    # Data files from thermochimica `data/` directory
    # See https://github.com/ornl-cees/thermochimica
    ("FeCuCbase.dat", 6, 4),
    ("FeTiVO.dat", 5, 21),
    ("Kaye_NobleMetals.dat", 9, 8),
    ("ZIRC-noSUBI.dat", 22, 28),
    ("test14.dat", 42, 8),

    # Data files from publications
]


@pytest.mark.parametrize("fn, num_soln_phases, num_stoich_phases", database_filenames)
def test_chemsage_reading(fn, num_soln_phases, num_stoich_phases):
    try:
        with open(fn) as fp:
            lines = fp.read()
            cs_dat_grammar = create_cs_dat_grammar()
            out = cs_dat_grammar.parseString(lines)
    except Exception as e:
        print('fail ' + fn + ' - ', end='')
        print(e)
        raise e
    assert len(out.solution_phases) == num_soln_phases
    assert len(out.stoichiometric_phases) == num_stoich_phases



header_data = [
    # filename, num_soln_phases, num_stoich_phases, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs

    # Highest priority to pass:
    ("CuZnFeCl-Viitala (1).dat", 5, 11, 8, 6, 6),  # https://doi.org/10.1016/j.calphad.2019.101667

    # Data files from FACT documentation
    # See https://gtt-technologies.de/software/chemapp/documentation/online-manual/
    ("Pb-Sn.dat", 5, 0, 2, 6, 2),
    ("C-N-O.dat", 1, 2, 3, 6, 4),
    ("C-O-Si.dat", 1, 7, 3, 6, 1),
    ("Fe-C.dat", 4, 2, 2, 6, 4),
    ("Fe2SiO4-Mg2SiO4.dat", 4, 2, 2, 6, 1),
    ("O-H-EA.dat", 2, 1, 3, 6, 6),
    ("Pitzer.dat", 2, 6, 6, 1, 1),
    ("subl-ex.dat", 5, 0, 3, 4, 4),

    # Data files from thermochimica `data/` directory
    # See https://github.com/ornl-cees/thermochimica
    ("C-O.dat", 1, 4, 2, 6, 6),
    ("W-Au-Ar-Ne-O_04.dat", 3, 5, 5, 6, 6),
    ("FeCuCbase.dat", 7, 4, 3, 6, 6),
    ("FeTiVO.dat", 5, 21, 4, 6, 6),
    ("Kaye_NobleMetals.dat", 9, 8, 4, 6, 6),
    ("ZIRC-noSUBI.dat", 22, 28, 9, 6, 6),
    ("test14.dat", 42, 8, 4, 6, 6),

    # Data files from publications
]
@pytest.mark.parametrize("fn, num_soln_phases, num_stoich_phases, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs", header_data)
def test_header_parsing(fn, num_soln_phases, num_stoich_phases, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs):
    with open(fn) as fp:
        lines = fp.read()
    out = grammar_header().parseString(lines)
    print(out)
    print(repr(out))
    assert len(out.list_soln_species_count) == num_soln_phases
    assert out.num_stoich_phases == num_stoich_phases
    assert len(out.pure_elements) == num_pure_elements
    assert len(out.pure_elements_mass) == num_pure_elements
    assert len(out.gibbs_coefficient_idxs) == num_gibbs_coeffs
    assert len(out.excess_coefficient_idxs) == num_excess_coeffs


def test_parse_viitala_header():
    HEADER = """ System Pb-Zn-Cu-Fe-Cl-e(CuCl)-e(FeZnsoln)-e(ZnFesoln)
    8    5    0   21    3    4    3   11
 Pb                       Zn                       Cu
 Fe                       Cl                       e(CuCl)
 e(FeZnsoln)              e(ZnFesoln)
   207.20000000              65.38000000              63.54600000
    55.84500000              35.45300000               0.00054858
     0.00054858               0.00054858
   6   1   2   3   4   5   6
   6   1   2   3   4   5   6

   adfa
   asdfl
   asdfjkasdf
   asdf
    """
    out = grammar_header().parseString(HEADER)
    print(out)
    assert len(out.list_soln_species_count) == 5
    assert out.list_soln_species_count.asList() == [0, 21, 3, 4, 3]
    assert out.num_stoich_phases == 11
    assert len(out.pure_elements) == 8
    assert out.pure_elements.asList() == ['Pb', 'Zn', 'Cu', 'Fe', 'Cl', 'e(CuCl)', 'e(FeZnsoln)', 'e(ZnFesoln)']
    assert np.allclose(out.pure_elements_mass.asList(), [207.2, 65.38, 63.546, 55.845, 35.453, 0.00054858, 0.00054858, 0.00054858])
    assert len(out.gibbs_coefficient_idxs) == 6
    assert out.gibbs_coefficient_idxs.asList() == [1, 2, 3, 4, 5, 6]
    assert len(out.excess_coefficient_idxs) == 6
    assert out.gibbs_coefficient_idxs.asList() == [1, 2, 3, 4, 5, 6]


def test_parse_endmember():
    ENDMEMBER_1 = """ CuCl
   1  1    0.0    0.0    1.0    0.0    1.0    0.0    0.0    0.0
  3000.0000     -151122.87      354.57317     -66.944000         0.00000000
     0.00000000     0.00000000
    """
    out = grammar_endmember(8, 6).parseString(ENDMEMBER_1)
    print(repr(out))
    print('------')
    print(out)
    assert out.name == 'CuCl'
    assert out.gibbs_eq_type == 1
    assert len(out.pair_stoichiometry) == 8
    assert len(out.intervals) == 1
    assert np.isclose(out.intervals[0].temperature, 3000.0)
    assert len(out.intervals[0].coefficients) == 6
    assert len(out.intervals[0].additional_coeff_pairs) == 0

    ENDMEMBER_2 = """ FeCl3
   4  3    0.0    0.0    0.0    1.0    3.0    0.0    0.0    0.0
  577.00000     -1419517.7      30687.563     -3436.4600     0.62998063
     0.00000000     0.00000000
 2  674287.33      99.00 -360918.86       0.50
  1500.0000      6608071.2     -33634.605      3322.5090     -.35542634
 0.15640951E-04 -87464397.
 2 -2322827.9      99.00  630846.86       0.50
  6000.0000     -370906.17      349.95802     -83.000000         0.00000000
     0.00000000     0.00000000
 1     0.00000000   0.00

    """
    out = grammar_endmember(8, 6).parseString(ENDMEMBER_2)
    print(repr(out))
    print('------')
    print(out)
    assert out.name == 'FeCl3'
    assert out.gibbs_eq_type == 4
    assert len(out.pair_stoichiometry) == 8
    assert len(out.intervals) == 3
    print()
    print(out.intervals[0].additional_coeff_pairs)
    print(repr(out.intervals[0].additional_coeff_pairs))
    assert np.isclose(out.intervals[0].temperature, 577.0)
    assert len(out.intervals[0].coefficients) == 6
    assert len(out.intervals[0].additional_coeff_pairs) == 2
    assert np.isclose(out.intervals[1].temperature, 1500.0)
    assert len(out.intervals[1].coefficients) == 6
    assert len(out.intervals[1].additional_coeff_pairs) == 2
    assert np.isclose(out.intervals[2].temperature, 6000.0)
    assert len(out.intervals[2].coefficients) == 6
    assert len(out.intervals[2].additional_coeff_pairs) == 1

