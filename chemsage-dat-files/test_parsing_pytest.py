# coding: utf-8
import pytest

import pyparsing
pyparsing.ParserElement.enablePackrat(0)
from pycalphad.io.cs_dat import create_cs_dat_grammar

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

