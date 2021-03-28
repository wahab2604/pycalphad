from pycalphad import variables as v

class ReferenceState():
    """
    Define the phase and any fixed state variables as a reference state for a component.

    Parameters
    ----------

    Attributes
    ----------
    fixed_statevars : dict
        Dictionary of {StateVariable: value} that will be fixed, e.g. {v.T: 298.15, v.P: 101325}
    phase_name : str
        Name of phase
    species : Species
        pycalphad Species variable

    """

    def __init__(self, species, reference_phase, fixed_statevars=None):
        """
        Parameters
        ----------
        species : str or Species
            Species to define the reference state for. Only pure elements supported.
        reference_phase : str
            Name of phase
        fixed_statevars : None, optional
            Dictionary of {StateVariable: value} that will be fixed, e.g. {v.T: 298.15, v.P: 101325}
            If None (the default), an empty dict will be created.

        """
        if isinstance(species, v.Species):
            self.species = species
        else:
            self.species = v.Species(species)
        self.phase_name = reference_phase
        self.fixed_statevars = fixed_statevars if fixed_statevars is not None else {}

    def __repr__(self):
        if len(self.fixed_statevars.keys()) > 0:
            s = "ReferenceState('{}', '{}', {})".format(self.species.name, self.phase_name, self.fixed_statevars)
        else:
            s = "ReferenceState('{}', '{}')".format(self.species.name, self.phase_name)
        return s
