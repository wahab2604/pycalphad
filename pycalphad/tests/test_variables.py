"""
Test variables module.
"""

import numpy as np
import pytest
from pycalphad import Database
from pycalphad import variables as v
from .datasets import CUO_TDB


SPECIES_EXAMPLES = [
    ("CL[-]", "CL[-]", {"CL": 1.0}, -1),
    ("CU[+]", "CU[+]", {"CU": 1.0}, 1),
    ("FE[3+]", "FE[3+]", {"FE": 1.0}, 3),
    ("FE1[2+]", "FE1[2+]", {"FE": 1.0}, 2),
    ("FE2[2+]", "FE2[2+]", {"FE": 2.0}, 2),
    ("ZN1CL3[1-]", "ZN1CL3[1-]", {"ZN": 1.0, "CL": 3.0}, -1),
    ("FE1O1.5", "FE1O1.5", {"FE": 1.0, "O": 1.5}, 0)
]


def test_species_parse_unicode_strings():
    """Species should properly parse unicode strings."""
    s = v.Species(u"MG")


@pytest.mark.parametrize("species_str, expected_name, expected_constituents, expected_charge", SPECIES_EXAMPLES)
def test_species_parsing(species_str, expected_name, expected_constituents, expected_charge):
    sp = v.Species(species_str)
    assert sp.name == expected_name
    assert set(sp.constituents.keys()) == set(expected_constituents.keys())
    for el in sp.constituents.keys():
        assert np.isclose(sp.constituents[el], expected_constituents[el])
    assert np.isclose(sp.charge, expected_charge)


def test_mole_and_mass_fraction_conversions():
    """Test mole <-> mass conversions work as expected."""
    # Passing database as a mass dict works
    dbf = Database(CUO_TDB)
    mole_fracs = {v.X('O'): 0.5}
    mass_fracs = v.get_mass_fractions(mole_fracs, v.Species('CU'), dbf)
    assert np.isclose(mass_fracs[v.W('O')], 0.20113144)  # TC
    # Conversion back works
    round_trip_mole_fracs = v.get_mole_fractions(mass_fracs, 'CU', dbf)
    assert all(np.isclose(round_trip_mole_fracs[mf], mole_fracs[mf]) for mf in round_trip_mole_fracs.keys())

    # Using Thermo-Calc's define components to define Al2O3 and TiO2
    # Mass dict defined by hand
    md = {'AL': 26.982, 'TI': 47.88, 'O': 15.999}
    alumina = v.Species('AL2O3')
    mass_fracs = {v.W(alumina): 0.81, v.W("TIO2"): 0.13}
    mole_fracs = v.get_mole_fractions(mass_fracs, 'O', md)
    assert np.isclose(mole_fracs[v.X('AL2O3')], 0.59632604)  # TC
    assert np.isclose(mole_fracs[v.X('TIO2')], 0.12216562)  # TC
    # Conversion back works
    round_trip_mass_fracs = v.get_mass_fractions(mole_fracs, v.Species('O'), md)
    assert all(np.isclose(round_trip_mass_fracs[mf], mass_fracs[mf]) for mf in round_trip_mass_fracs.keys())
