# coding: utf-8
import pytest

import numpy as np
from pycalphad.io.cs_dat import *

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
    out = parse_header(tokenize(lines, 1))
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
    alltoks = tokenize(HEADER, 1)
    print(alltoks)
    out = parse_header(alltoks)
    print(out)
    assert len(out.list_soln_species_count) == 5
    assert out.list_soln_species_count == [0, 21, 3, 4, 3]
    assert out.num_stoich_phases == 11
    assert len(out.pure_elements) == 8
    assert out.pure_elements == ['Pb', 'Zn', 'Cu', 'Fe', 'Cl', 'e(CuCl)', 'e(FeZnsoln)', 'e(ZnFesoln)']
    assert np.allclose(out.pure_elements_mass, [207.2, 65.38, 63.546, 55.845, 35.453, 0.00054858, 0.00054858, 0.00054858])
    assert len(out.gibbs_coefficient_idxs) == 6
    assert out.gibbs_coefficient_idxs == [1, 2, 3, 4, 5, 6]
    assert len(out.excess_coefficient_idxs) == 6
    assert out.gibbs_coefficient_idxs == [1, 2, 3, 4, 5, 6]


def test_parse_endmember():
    ENDMEMBER_1 = tokenize(""" CuCl
   1  1    0.0    0.0    1.0    0.0    1.0    0.0    0.0    0.0
  3000.0000     -151122.87      354.57317     -66.944000         0.00000000
     0.00000000     0.00000000
    """)
    out = parse_endmember(ENDMEMBER_1, 8, 6)
    print(repr(out))
    print('------')
    print(out)
    assert out.species_name == 'CuCl'
    assert out.gibbs_eq_type == 1
    assert len(out.stoichiometry_pure_elements) == 8
    assert len(out.intervals) == 1
    assert np.isclose(out.intervals[0].temperature, 3000.0)
    assert len(out.intervals[0].coefficients) == 6
    assert len(out.intervals[0].additional_coeff_pairs) == 0

    ENDMEMBER_2 = tokenize(""" FeCl3
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
    """)
    out = parse_endmember(ENDMEMBER_2, 8, 6)
    print(repr(out))
    print('------')
    print(out)
    assert out.species_name == 'FeCl3'
    assert out.gibbs_eq_type == 4
    assert len(out.stoichiometry_pure_elements) == 8
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


SUBQ_Viitala = """ Liquidsoln
 SUBQ
   6   6
 CuCl
   1  1    0.0    0.0    1.0    0.0    1.0    0.0    0.0    0.0
  3000.0000     -151122.87      354.57317     -66.944000         0.00000000
     0.00000000     0.00000000
  1.00000      1.00000         0.000000     0.000000     0.000000
  2.40000
 ZnCl2
   1  2    0.0    1.0    0.0    0.0    2.0    0.0    0.0    0.0
  1005.0000     -416867.43      308.48717     -65.835000     -.11564500E-01
     0.00000000     0.00000000
  3000.0000     -428547.87      469.17142     -89.079645         0.00000000
     0.00000000     0.00000000
  1.00000      2.00000         0.000000     0.000000     0.000000
  2.40000
 FeCl3
   1  2    0.0    0.0    0.0    1.0    3.0    0.0    0.0    0.0
  1500.0000     -402636.55      696.05336     -133.88800         0.00000000
     0.00000000     0.00000000
  3000.0000     -326304.55      273.01020     -83.000000         0.00000000
     0.00000000     0.00000000
  1.00000      3.00000         0.000000     0.000000     0.000000
  2.40000
 FeCl2
   1  2    0.0    0.0    0.0    1.0    2.0    0.0    0.0    0.0
  2000.0000     -341793.75      544.42204     -102.17300         0.00000000
     0.00000000     0.00000000
  3000.0000     -267447.75      224.70070     -65.000000         0.00000000
     0.00000000     0.00000000
  1.00000      2.00000         0.000000     0.000000     0.000000
  2.40000
 CuCl2
   4  2    0.0    0.0    1.0    0.0    2.0    0.0    0.0    0.0
  1500.0000     -174349.37      412.11241     -82.501348     -.16813021E-02
     0.00000000  1945.9013
 1 -3462.1777      99.00
  3000.0000     -196522.43      427.48835     -85.235406         0.00000000
     0.00000000     0.00000000
 1     0.00000000   0.00
  1.00000      2.00000         0.000000     0.000000     0.000000
  2.40000
 PbCl2
   1  3    1.0    0.0    0.0    0.0    2.0    0.0    0.0    0.0
  500.00000     -365945.98      313.37200     -68.390656     -.14604471E-01
     0.00000000     0.00000000
  2000.0000     -383851.53      609.81341     -111.50400         0.00000000
     0.00000000     0.00000000
  3000.0000     -280843.53      166.83253     -60.000000         0.00000000
     0.00000000     0.00000000
  1.00000      2.00000         0.000000     0.000000     0.000000
  2.40000
   6   1
 Cu                       Zn                       Fe
 Cu                       Fe                       Pb
 Cl
  1.00000      2.00000      3.00000      2.00000      2.00000      2.00000
   1   1   1   1   1   1
  1.00000
   1
   1   2   3   5   4   6
   1   1   1   1   1   1
   1   1   7   7  6.0000000      6.0000000      6.0000000      6.0000000
   2   2   7   7  6.0000000      6.0000000      3.0000000      3.0000000
   3   3   7   7  6.0000000      6.0000000      2.0000000      2.0000000
   5   5   7   7  6.0000000      6.0000000      3.0000000      3.0000000
   4   4   7   7  6.0000000      6.0000000      3.0000000      3.0000000
   6   6   7   7  6.0000000      6.0000000      3.0000000      3.0000000
   3
 G   1   2   7   7   0   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0  3960.6600     -3.7732600         0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   1   3   7   7   0   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -4364.0000         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   1   2   7   7   1   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -2018.2700         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   1   2   7   7   0   1   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -4457.2100     -.78300000         0.00000000     0.00000000
     0.00000000     0.00000000
   3
 Q   1   3   7   7   1   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0  2940.0000      3.2400000         0.00000000     0.00000000
     0.00000000     0.00000000
   3
 Q   1   3   7   7   0   1   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -1534.0000         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   3   5   7   7   0   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -4855.0000      4.1900000         0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   2   3   7   7   0   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -3865.0000         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   2   3   7   7   1   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -5687.0000         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   2   3   7   7   0   1   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0  634.00000         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   3
 Q   4   5   7   7   0   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -2191.3800         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   1   6   7   7   0   0   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -360.80000         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   3
 G   1   6   7   7   0   1   0   0
     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00
     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00
   0   0 -798.80000         0.00000000     0.00000000     0.00000000
     0.00000000     0.00000000
   0
"""

def test_parse_subq_phase():
    toks = tokenize(SUBQ_Viitala)
    phase_subq = parse_phase(toks, 8, 6)
    assert len(phase_subq.endmembers) == 6
    for em in phase_subq.endmembers:
        assert len(em.stoichiometry_quadruplet) == 5
        assert np.isclose(em.coordination, 2.4)
    assert phase_subq.num_subl_1_const == 6
    assert phase_subq.num_subl_2_const == 1
    assert phase_subq.subl_1_const == ['Cu', 'Zn', 'Fe', 'Cu', 'Fe', 'Pb']
    assert phase_subq.subl_2_const == ['Cl']
    assert phase_subq.subl_1_chemical_groups == [1, 1, 1, 1, 1, 1]
    assert phase_subq.subl_2_chemical_groups == [1]
    assert np.allclose(phase_subq.subl_1_charges, [1.0, 2.0, 3.0, 2.0, 2.0, 2.0])
    assert np.allclose(phase_subq.subl_2_charges, [1.0])
