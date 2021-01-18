# coding: utf-8
import pytest

import numpy as np
from pycalphad.io.cs_dat import *

header_data = [
    # filename, num_soln_phases, num_stoich_phases, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs

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
    ("CuZnFeCl-Viitala (1).dat", 5, 11, 8, 6, 6),  # https://doi.org/10.1016/j.calphad.2019.101667
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
    assert np.isclose(out.intervals[0].T_max, 3000.0)
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
    assert np.isclose(out.intervals[0].T_max, 577.0)
    assert len(out.intervals[0].coefficients) == 6
    assert len(out.intervals[0].additional_coeff_pairs) == 2
    assert np.isclose(out.intervals[1].T_max, 1500.0)
    assert len(out.intervals[1].coefficients) == 6
    assert len(out.intervals[1].additional_coeff_pairs) == 2
    assert np.isclose(out.intervals[2].T_max, 6000.0)
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
    # number of constituents (last number) doesn't matter for SUBQ
    phase_subq = parse_phase(toks, 8, 6, 6, 0)
    assert len(toks) == 0  # completed parsing
    assert len(phase_subq.endmembers) == 6
    for em in phase_subq.endmembers:
        assert len(em.stoichiometry_quadruplet) == 5
        assert np.isclose(em.zeta, 2.4)
    assert phase_subq.num_quadruplets == 6
    assert phase_subq.num_subl_1_const == 6
    assert phase_subq.num_subl_2_const == 1
    assert phase_subq.subl_1_const == ['Cu', 'Zn', 'Fe', 'Cu', 'Fe', 'Pb']
    assert phase_subq.subl_2_const == ['Cl']
    assert phase_subq.subl_1_chemical_groups == [1, 1, 1, 1, 1, 1]
    assert phase_subq.subl_2_chemical_groups == [1]
    assert np.allclose(phase_subq.subl_1_charges, [1.0, 2.0, 3.0, 2.0, 2.0, 2.0])
    assert np.allclose(phase_subq.subl_2_charges, [1.0])
    assert phase_subq.subl_const_idx_pairs == [(1, 1), (2, 1), (3, 1), (5, 1), (4, 1), (6, 1)]
    assert len(phase_subq.quadruplets) == phase_subq.num_quadruplets
    assert phase_subq.quadruplets[2] == Quadruplet([3, 3, 7, 7], [6.0, 6.0, 2.0, 2.0])
    num_excess_G = len([x for x in phase_subq.excess_parameters if x.mixing_code == 'G'])
    assert num_excess_G == 10
    num_excess_Q = len([x for x in phase_subq.excess_parameters if x.mixing_code == 'Q'])
    assert num_excess_Q == 3


IDMX_CO = """ gas_ideal
 IDMX
 C
   4  3    0.0    1.0
  2600.0000      712234.84     -11.891684     -21.741391     0.47346994E-03
 -.53611800E-07 -28855.579
 1 -386.20250      99.00
  6000.0000      1207023.0     -98.547935     -15.608903     0.00000000
 0.00000000     -29392513.
 2  6562.6344       0.50 -91434.292      99.00
  6001.0000      701947.43      2.8618228     -23.183583     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 C2
   4  3    0.0    2.0
  800.00000      871764.76      303.15125     -61.761335     0.00000000
 0.00000000     -3625790.5
 2 -3989.6220       0.50 0.18740113E+09  -2.00
  6000.0000      660845.42      577.82306     -86.913591     0.18602205E-02
 -.55497861E-07  2311422.7
 2  50919.010      99.00 -11873.025       0.50
  6001.0000      792616.68      135.07245     -46.616551     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 C3
   4  3    0.0    3.0
  1500.0000      723423.86      845.05778     -121.92070     0.38058599E-02
 0.00000000      543160.92
 2  36528.039      99.00 -13274.987       0.50
  6000.0000      785920.96      310.21610     -68.644142     0.00000000
 0.00000000      1363274.3
 2  8705.9429      99.00 -3803.8336       0.50
  6001.0000      771494.48      181.33567     -57.742574     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 C4
   4  3    0.0    4.0
  1400.0000      922816.78      1115.1861     -160.26358     0.54306211E-02
 0.00000000      344608.30
 2  21259.177      99.00 -11768.944       0.50
  6000.0000      977846.25      372.38730     -85.937369     0.00000000
 0.00000000      2832263.4
 2 -11212.104      99.00  930.64079       0.50
  6001.0000      922640.55      385.92943     -86.914967     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 C5
   4  3    0.0    5.0
  1400.0000      917676.53      1496.0821     -206.81334     0.69790477E-02
 0.00000000      532674.46
 2  27433.257      99.00 -15301.437       0.50
  6000.0000      992263.13      530.96948     -110.35172     0.00000000
 0.00000000      3724647.1
 2 -15513.530      99.00  1289.6211       0.50
  6001.0000      915786.51      549.89052     -111.72145     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 O
   4  3    1.0    0.0
  2200.0000      242682.20     -30.020448     -19.964968     -.13194946E-03
 0.61589319E-08 -29327.439
 1  83.809856       0.50
  6000.0000     -84471.471      186.78944     -38.050669     0.00000000
 0.00000000      12120739.
 2  70811.436      99.00 -8330.8535       0.50
  6001.0000      236685.00     -8.3771889     -22.291487     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 O2
   4  4    2.0    0.0
  1000.0000     -5219.3324     -12.179856     -26.924057     -.84893408E-02
 0.11276942E-05 -114664.63
 1 -316.64663       0.50
  4000.0000     -389938.78      638.67678     -89.681327     0.72372244E-03
 0.00000000      9341343.0
 2 -16506.149       0.50  95803.960      99.00
  6000.0000     -8951197.1      2742.7612     -249.17312     0.00000000
 0.00000000     0.59248921E+09
 2 -139742.68       0.50  1674792.2      99.00
  6001.0000     -42014.302      116.92408     -44.371534     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 O3
   4  3    3.0    0.0
  1900.0000      200310.28      315.13014     -76.555764     0.46115697E-02
 -.32267167E-06 -821844.45
 1 -15871.007      99.00
  6000.0000     -95706.532      344.94952     -73.713624     0.00000000
 0.00000000      9320171.5
 2  47079.583      99.00 -6138.3410       0.50
  6001.0000      107048.81      191.83924     -61.231023     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 CO
   4  3    1.0    1.0
  1700.0000     -203680.54      609.35468     -90.753584     0.31175713E-02
 0.00000000      770627.67
 2  32063.316      99.00 -10357.020       0.50
  6000.0000     -224220.54      142.21490     -44.086320     0.00000000
 0.00000000      5816543.2
 2  20893.908      99.00 -2749.1074       0.50
  6001.0000     -133600.03      72.371771     -38.372791     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 C2O
   4  3    1.0    2.0
  2300.0000      231186.62      664.35777     -107.49204     0.18364932E-02
 0.00000000      413854.90
 2  19637.467      99.00 -8280.8929       0.50
  6000.0000      276457.54      102.96587     -54.804651     0.00000000
 0.00000000      10638866.
 2 -14952.771      99.00  3832.3311       0.50
  6001.0000      257586.58      214.98181     -64.090270     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 CO2
   4  3    2.0    1.0
  1900.0000     -415578.81      642.65849     -103.34460     0.23713031E-02
 0.00000000      20124.522
 2 -6993.1489       0.50  11004.741      99.00
  6000.0000      402917.74     -923.17324      49.197241     0.00000000
 0.00000000     -21647150.
 3 -.82448878       1.50  30319.476       0.50 -196911.68      99.00
  6001.0000     -439389.21      251.60871     -64.940924     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
 C3O2
   4  3    2.0    3.0
  1400.0000     -233348.58      1753.8120     -234.75261     0.83038458E-02
 0.00000000      1214665.1
 2  53046.991      99.00 -21646.368       0.50
  6000.0000     -54630.888      487.70053     -109.73369     0.00000000
 0.00000000      2913316.3
 2 -20511.152      99.00  1708.0096       0.50
  6001.0000     -157027.98      513.96765     -111.66589     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
"""


def test_parse_idmx():
    toks = tokenize(IDMX_CO)
    phase_cef = parse_phase(toks, 2, 6, 6, 12)
    assert len(phase_cef.endmembers) == 12
    assert phase_cef.endmembers[-1].species_name == 'C3O2'
    assert np.allclose(phase_cef.endmembers[-1].stoichiometry_pure_elements, [2.0, 3.0])
    assert len(toks) == 0  # completion


STOICH_C_diamond = """ C_diamond(s2)
   4  2    0.0    1.0
  3000.0000      30879.186      209.52835     -28.008000     -.21380000E-03
 0.00000000     -155590.00
 1 -7729.5000      99.00
  3001.0000     -17678.685      194.34620     -26.748876     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
"""


def test_parse_stoich_phase():
    toks = tokenize(STOICH_C_diamond)
    phase_stoich = parse_stoich_phase(toks, 2, 6)
    assert phase_stoich.phase_name == 'C_diamond(s2)'
    assert len(phase_stoich.endmembers) == 1
    assert len(phase_stoich.endmembers[0].intervals) == 2
    assert len(toks) == 0  # completion


# determining tokens remaining:
# def remtoks(filename):
#     with open(filename) as fp:
#         data = fp.read()
#     toks = tokenize(data, 1)
#     idxs = [i for i, x in enumerate(toks) if x.startswith('##')]
#     print(len(toks) - idxs[0] if len(idxs) > 0 else 0)


full_parses = [
    # filename, num_soln_phases, num_stoich_phases, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs

    # Data files from FACT documentation
    # See https://gtt-technologies.de/software/chemapp/documentation/online-manual/
    ("Pb-Sn.dat", 90),
    ("C-N-O.dat", 66),
    ("C-O-Si.dat", 113),
    ("Fe-C.dat", 308),
    ("Fe2SiO4-Mg2SiO4.dat", 142),
    ("O-H-EA.dat", 102),
    ("Pitzer.dat", 94),
    ("subl-ex.dat", 181),

    # Data files from thermochimica `data/` directory
    # See https://github.com/ornl-cees/thermochimica
    ("C-O.dat", 0),
    ("W-Au-Ar-Ne-O_04.dat", 0),
    ("FeCuCbase.dat", 0),
    ("FeTiVO.dat", 0),
    ("Kaye_NobleMetals.dat", 0),
    ("ZIRC-noSUBI.dat", 0),
    ("test14.dat", 0),

    # Data files from publications
    ("CuZnFeCl-Viitala (1).dat", 0),
]


@pytest.mark.parametrize("filename, remaining_toks", full_parses)
def test_full_parsing(filename, remaining_toks):
    with open(filename) as fp:
        data = fp.read()
    header, solns, stoichs, toks = parse_cs_dat(data)
    assert len(toks) == remaining_toks  # some files have comments, etc. at the end. this accounts for that.


# 2 species interacting, 4 coeffs
EXCESS_TERM_1 = """ 2
 1   2   3
-124320.00      28.500000     0.00000000     0.00000000
 19300.000     0.00000000     0.00000000     0.00000000
 49260.000     -19.000000     0.00000000     0.00000000
 2
 1   3   1
-34671.000     0.00000000     0.00000000     0.00000000
 0
"""


def test_parse_excess_parameters():
    toks = tokenize(EXCESS_TERM_1)
    excess_terms = parse_excess_parameters(toks, 4)
    assert len(toks) == 0
    assert excess_terms[0].interacting_species_idxs == [1, 2]
    assert excess_terms[1].interacting_species_idxs == [1, 2]
    assert excess_terms[2].interacting_species_idxs == [1, 2]
    assert excess_terms[3].interacting_species_idxs == [1, 3]
    assert len(excess_terms) == 4
    assert np.allclose(excess_terms[1].coefficients, [19300.0, 0.0, 0.0, 0.0])
    assert np.allclose(excess_terms[3].coefficients, [-34671.0, 0.0, 0.0, 0.0])
    assert [len(xt.coefficients) for xt in excess_terms] == [4, 4, 4, 4]


def test_gibbs_interval_construction():
    interval_str = "577.00000 -1419517.7 30687.563 -3436.4600 0.62998063 0.00000000 0.00000000 2 674287.33 99.00 -360918.86 0.50"
    EXPECTED_EXPR = -1419517.7 + 30687.563*v.T - 3436.4600*v.T*log(v.T) + 0.62998063*v.T**2 + 674287.33*v.T**(99.00) - 360918.86*v.T**(0.50)
    EXPECTED_COND = (298.15 <= v.T) & (v.T < 577.00000)
    toks = tokenize(interval_str)
    interval = parse_interval_Gibbs(toks, 6, has_additional_terms=True, has_PTVm_terms=False)
    expr = interval.expr([1, 2, 3, 4, 5, 6])
    assert expr == EXPECTED_EXPR
    cond = interval.cond(T_min=298.15)
    assert len(cond.args) == 2
    assert cond == EXPECTED_COND
    expr2, cond2 = interval.expr_cond_pair([1, 2, 3, 4, 5, 6], T_min=298.15)
    assert expr2 == expr
    assert cond2 == cond


def test_endmember_expression_construction_RKMP():
    em_str = """ C
   4  3    0.0    1.0
  2600.0000      712234.84     -11.891684     -21.741391     0.47346994E-03
 -.53611800E-07 -28855.579
 1 -386.20250      99.00
  6000.0000      1207023.0     -98.547935     -15.608903     0.00000000
 0.00000000     -29392513.
 2  6562.6344       0.50 -91434.292      99.00
  6001.0000      701947.43      2.8618228     -23.183583     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
    """
    toks = tokenize(em_str, force_upper=True)
    em = parse_endmember(toks, 2, 6)
    expr = em.expr([1, 2, 3, 4, 5, 6])
    assert isinstance(expr, Piecewise)
    assert len(expr.as_expr_set_pairs()) == 3
    assert em.constituent_array() == [['C']]
    assert len(em.species(['FE', 'C'])) == 1


def test_endmember_expression_construction_SUBL():
    em_str = """ Co:Cr:Co
   4  3    0.866666667    0.133333333
  2600.0000      712234.84     -11.891684     -21.741391     0.47346994E-03
 -.53611800E-07 -28855.579
 1 -386.20250      99.00
  6000.0000      1207023.0     -98.547935     -15.608903     0.00000000
 0.00000000     -29392513.
 2  6562.6344       0.50 -91434.292      99.00
  6001.0000      701947.43      2.8618228     -23.183583     0.00000000
 0.00000000     0.00000000
 1 0.00000000       0.00
    """
    toks = tokenize(em_str, force_upper=True)
    em = parse_endmember(toks, 2, 6)
    expr = em.expr([1, 2, 3, 4, 5, 6])
    assert isinstance(expr, Piecewise)
    assert len(expr.as_expr_set_pairs()) == 3
    assert em.constituent_array() == [['CO'], ['CR'], ['CO']]
    assert len(em.species(['CO', 'CR'])) == 2


full_parses = [
    # filename, num_soln_phases, num_stoich_phases, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs

    # Data files from FACT documentation
    # See https://gtt-technologies.de/software/chemapp/documentation/online-manual/
    ("Pb-Sn.dat", ''),
    ("C-N-O.dat", ''),
    ("C-O-Si.dat", ''),
    ("Fe-C.dat", ''),
    ("Fe2SiO4-Mg2SiO4.dat", ''),
    pytest.param("O-H-EA.dat", '', marks=pytest.mark.xfail),  # Real Gas model not supported
    pytest.param("Pitzer.dat", '', marks=pytest.mark.xfail),  # Pitzer model not implemented
    ("subl-ex.dat", ''),

    # Data files from thermochimica `data/` directory
    # See https://github.com/ornl-cees/thermochimica
    ("C-O.dat", ''),
    ("W-Au-Ar-Ne-O_04.dat", ''),
    ("FeCuCbase.dat", ''),
    ("FeTiVO.dat", ''),
    ("Kaye_NobleMetals.dat", ''),
    ("ZIRC-noSUBI.dat", ''),
    ("test14.dat", ''),

    # Data files from publications
    ("CuZnFeCl-Viitala (1).dat", ''),
    ("Pb-Zn-Cu-Fe-Cl-stoich-only.dat", ''),
]


@pytest.mark.parametrize("filename, _", full_parses)
def test_reading_chemsage_databases(filename, _):
    dbf = Database(filename)
