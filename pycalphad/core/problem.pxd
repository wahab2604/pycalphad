from pycalphad.core.composition_set cimport CompositionSet

cdef class Problem:
    cdef public int num_constraints
    cdef public object composition_sets
    cdef public object conditions
    cdef public object components
    cdef public int num_phases
    cdef public int num_vars
    cdef public double temperature
    cdef public double pressure
    cdef public double[::1] xl
    cdef public double[::1] xu
    cdef public double[::1] x0
    cdef public double[::1] cl
    cdef public double[::1] cu

cdef class FixedChemicalPotentialProblem:
    cdef public int num_constraints
    cdef public CompositionSet compset
    cdef public double[::1] chemical_potentials
    cdef public object conditions
    cdef public object components
    cdef public int num_vars
    cdef public double temperature
    cdef public double pressure
    cdef public double[::1] xl
    cdef public double[::1] xu
    cdef public double[::1] x0
    cdef public double[::1] cl
    cdef public double[::1] cu