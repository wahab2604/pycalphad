import pytest

import numpy as np
import pyparsing
pyparsing.ParserElement.enablePackrat(0)
from pycalphad.io.cs_dat import create_cs_dat_grammar, grammar_header, parse_cs_dat


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


def test_parse_viitala_header():
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


    assert False
    # num_pure_elements = out[0]
    # soln_phase_species_counts = out[1]
    # num_stoichiometric_phases = out[2]
    # assert num_pure_elements == 8
    # assert soln_phase_species_counts == [0, 21, 3, 4, 3]
    # assert num_stoichiometric_phases == 11
