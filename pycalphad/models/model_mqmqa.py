import itertools
from collections import Counter
from functools import partial
from sympy import S, log, Piecewise, And
from sympy import exp, log, Abs, Add, And, Float, Mul, Piecewise, Pow, S, sin, StrictGreaterThan, Symbol, zoo, oo, nan
from tinydb import where
from pycalphad.core.utils import unpack_components
from pycalphad import Model
from pycalphad.io.cs_dat import get_species
from pycalphad.core.constraints import is_multiphase_constraint


class ModelMQMQA(Model):
    """

    One peculiarity about the ModelMQMQA is that the charges in the way the
    model are written are assumed to be positive. We take the absolute value
    whenever there is a charge.

    """

    contributions = [
        ('ref', 'traditional_reference_energy'),
        ('idmix', 'ideal_mixing_energy'),
        ('xsmix', 'excess_mixing_energy'),
    ]

    def __init__(self, dbe, comps, phase_name, parameters=None):
        # Here we do some custom initialization before calling
        # `Model.__init__` via `super()`, which does the initialization and
        # builds the phase as usual.

        # build `constituents` here so we can build the pairs and quadruplets
        # *before* `super().__init__` calls `self.build_phase`. We leave it to
        # the Model to build self.constituents and do the error checking.
        active_species = unpack_components(dbe, comps)
        constituents = []
        for sublattice in dbe.phases[phase_name].constituents:
            sublattice_comps = set(sublattice).intersection(active_species)
            constituents.append(sublattice_comps)

        # create self.cations and self.anions properties to use instead of constituents
        self.cations = sorted(constituents[0])
        self.anions = sorted(constituents[1])

        # Call Model.__init__, which will build the Gibbs energy from the contributions list.
        super().__init__(dbe, comps, phase_name, parameters=parameters)

        # In several places we use the assumption that the cation lattice and anion lattice have no common species
        # we validate that assumption here
        shared_species = set(self.cations).intersection(set(self.anions))
        assert len(shared_species) == 0, f"No species can be shared between the two MQMQA lattices, got {shared_species}"

        # pycalphad now expects the `constituents` to refer to the constituents w.r.t. the Gibbs energy and internal DOF, not the phase constituents
        # we fix that here.
        quads = itertools.product(itertools.combinations_with_replacement(self.cations, 2), itertools.combinations_with_replacement(self.anions, 2))
        quad_species = [get_species(A,B,X,Y) for (A, B), (X, Y) in quads]
        self.constituents = [sorted(quad_species)]

    def _p(self, *ABXYs: v.Species) -> v.SiteFraction:
        """Shorthand for creating a site fraction object v.Y for a quadruplet.

        The name `p` is intended to mirror construction of `p(A,B,X,Y)`
        quadruplets, following Sundman's notation.
        """
        return v.Y(self.phase_name, 0, get_species(*ABXYs))

    def _pair_test(self, constituent_array):
        """Return True if the constituent array represents a pair.

        Pairs have only one species in each sublattice.
        """
        for subl in constituent_array:
            if len(subl) > 1:
                return False
            constituent = subl[0]
            if constituent not in self.components:
                return False
        return True

    def _mixing_test(self, constituent_array):
        """Return True if the constituent array is satisfies all components.
                """
        for subl in constituent_array:
            for constituent in subl:
                if constituent not in self.components:
                    return False
        return True

    def M(self, dbe, species):
        """Return the mass of the species.

        The returned expression is composed only of v.Y objects for
        quadruplets, p(A,B,X,Y) in Sundman's notation. The expression
        constructed here follows equation (8) of Sundman's notes.

        This is the same as X_A in Pelton's notation.
        """
        cations = self.cations
        anions = self.anions

        # aliases for notation
        Z = partial(self.Z, dbe)
        p = self._p

        M = S.Zero
        if species in cations:
            A = species
            for i, X in enumerate(anions):
                for Y in anions[i:]:
                    M += p(A,A,X,Y)/Z(A,A,A,X,Y)
                    for B in cations:
                        M += p(A, B, X, Y)/Z(A, A, B, X, Y)
        else:
            assert species in anions
            X = species
            for i, A in enumerate(cations):
                for B in cations[i:]:
                    M += p(A,B,X,X)/Z(X,A,B,X,X)
                    for Y in anions:
                        M += p(A, B, X, Y)/Z(X, A, B, X, Y)
        return M

    def ξ(self, A: v.Species, X: v.Species):
        """Return the endmember fraction, ξ_A:X, for a pair Species A:X

        The returned expression is composed only of v.Y objects for
        quadruplets, p(A,B,X,Y) in Sundman's notation. The expression
        constructed here follow equation (12) of Sundman's notes.

        This is the same as X_A/X in Pelton's notation.
        """
        cations = self.cations
        anions = self.anions

        p = self._p  # alias to keep the notation close to the math

        # Sundman notes equation (12)
        return 0.25 * (
            p(A,A,X,X) +
        sum(p(A,A,X,Y) for Y in anions) +
        sum(p(A,B,X,X) for B in cations) +
        sum(p(A,B,X,Y) for B, Y in itertools.product(cations, anions))
        )

    def w(self, species: v.Species):
        """Return the coordination equivalent site fraction of species.

        The returned expression is composed only of v.Y objects for
        quadruplets, p(A,B,X,Y) in Sundman's notation. The expression
        constructed here follow equation (15) of Sundman's notes.


        This is the same as Y_i in Pelton's notation.
        """
        p = self._p
        cations = self.cations
        anions = self.anions

        w = S.Zero
        if species in cations:
            A = species
            for i, X in enumerate(anions):
                for Y in anions[i:]:
                    w += p(A, A, X, Y)
                    for B in cations:
                        w += p(A, B, X, Y)
        else:
            assert species in anions
            X = species
            for i, A in enumerate(cations):
                for B in cations[i:]:
                    w += p(A, B, X, X)
                    for Y in anions:
                        w += p(A, B, X, Y)
        return 0.5*w

    def ϑ(self, dbe, species: v.Species):
        """Return the site fraction of species on it's sublattice.

        The returned expression is composed only of v.Y objects for
        quadruplets, p(A,B,X,Y) in Sundman's notation, and (constant)
        coordination numbers. The expression constructed here follow equation
        (10) of Sundman's notes.

        This is the same as X_i in Pelton's notation.
        """
        cations = self.cations
        anions = self.anions

        if species in cations:
            return self.M(dbe, species)/sum(self.M(dbe, sp) for sp in cations)
        else:
            assert species in anions
            return self.M(dbe, species)/sum(self.M(dbe, sp) for sp in anions)

    def _calc_Z(self, dbe: Database, species, A, B, X, Y):
        Z = partial(self.Z,  dbe)
#         print(f'calculating $Z^{{{species}}}_{{{A} {B} {X} {Y}}}$')
        if (species == A) or (species == B):
            species_is_cation = True
        elif (species == X) or (species == Y):
            species_is_cation = False
        else:
            raise ValueError(f"{species} is not A ({A}), B ({B}), X ({X}) or Y ({Y}).")

        if A == B and X == Y:
            raise ValueError(f'Z({species}, {A}{B}/{X}{Y}) is a pure pair and must be defined explictly')
        elif A != B and X != Y:
            # This is a reciprocal AB/XY quadruplet and needs to be calculated by eq 23 and 24 in Pelton et al. Met Trans B (2001)
            F = 1/8 * (  # eq. 24
                  abs(A.charge)/Z(A, A, A, X, Y)
                + abs(B.charge)/Z(B, B, B, X, Y)
                + abs(X.charge)/Z(X, A, B, X, X)
                + abs(Y.charge)/Z(Y, A, B, Y, Y)
                )
            if species_is_cation:
                inv_Z = F * (
                              Z(X, A, B, X, X)/(abs(X.charge) * Z(species, A, B, X, X))
                            + Z(Y, A, B, Y, Y)/(abs(Y.charge) * Z(species, A, B, Y, Y))
                            )
            else:
                inv_Z = F * (
                              Z(A, A, A, X, Y)/(abs(A.charge) * Z(species, A, A, X, Y))
                            + Z(B, B, B, X, Y)/(abs(B.charge) * Z(species, B, B, X, Y))
                            )
            return 1/inv_Z
        elif A != B:  # X == Y
            # Need to calculate Z^i_AB/XX (Y = X).
            # We assume Z^A_ABXX = Z^A_AAXX = Z^A_AAYY
            # and Z^X_ABXX = (q_X + q_Y)/(q_A/Z^A_AAXX + q_B/Z^B_BBXX)  # note: q_X = q_Y, etc. since Y = X
            # We don't know if these are correct, but that's what's implemented in Thermochimica
            if species_is_cation:
                return Z(species, species, species, X, X)
            else:
#                 print(f'calculating bad $Z^{{{species}}}_{{{A} {B} {X} {Y}}}$')
                return 2*abs(species.charge)/(abs(A.charge)/Z(A, A, A, species, species) + abs(B.charge)/Z(B, B, B, species, species))
        elif X != Y:  # A == B
            # These use the same equations as A != B case with the same assumptions
            if species_is_cation:
                # similarly, Z^A_AAXY = (q_A + q_B)/(q_X/Z^X_AAXX + q_Y/Z^Y_AAYY)
#                 print(f'calculating bad $Z^{{{species}}}_{{{A} {B} {X} {Y}}}$')
                return 2*abs(species.charge)/(abs(X.charge)/Z(X, species, species, X, X) + abs(Y.charge)/Z(Y, species, species, Y, Y))
            else:
                return Z(species, A, A, species, species)
        raise ValueError("This should be unreachable")


    def Z(self, dbe: Database, species: v.Species, A: v.Species, B: v.Species, X: v.Species, Y: v.Species):
        Zs = dbe._parameters.search(
            (where('phase_name') == self.phase_name) & \
            (where('parameter_type') == "Z") & \
            (where('diffusing_species').test(lambda sp: sp.name == species.name)) & \
            # quadruplet needs to be in 1 sublattice constituent array `[[q]]`, in tuples
            (where('constituent_array').test(lambda x: x == ((A, B), (X, Y)) ) )
        )
        if len(Zs) == 0:
            # TODO: add this to the database so we don't need to recalculate? where should that happen?
            return self._calc_Z(dbe, species, A, B, X, Y)
        elif len(Zs) == 1:
            return Zs[0]['parameter']
        else:
            raise ValueError(f"Expected exactly one Z for {species} of {((A, B), (X, Y))}, got {len(Zs)}")

    def get_internal_constraints(self):
        constraints = []
        p = self._p
        total_quad = -1
        cations = self.cations
        anions = self.anions
        for i, A in enumerate(cations):
            for B in cations[i:]:
                for j, X in enumerate(anions):
                    for Y in anions[j:]:
                        total_quad += p(A,B,X,Y)
        constraints.append(total_quad)
        return constraints

    def moles(self, species):
        "Number of moles of species or elements."
        species = v.Species(species)
        result = S.Zero
        for i in itertools.chain(self.cations, self.anions):
            if list(species.constituents.keys())[0] in i.constituents:
                result += self.M(self._dbe, i)
        # moles is supposed to compute the moles of a pure element, but with a caveat that pycalphad assumes sum(moles(c) for c in comps) == 1
        # The correct solution is to make the changes where pycalphad assumes n=1. But I think it would be easier to change how we implement the model so that the model has n=1 and the energies are normalized to per-mole-atoms.
        # Since normalizing to moles of quadruplets is allowing us to easily compare with thermochimica, I'm thinking that we might be able to fake pycalphad into thinking we have N=1 by normalizing "moles" to n=1
        # The energies will not be normalized to moles of atoms (and so you cannot yet use this Model to compare to other phases), but internally it should be correct and in agreement with thermochimica

        normalization = sum(self.M(self._dbe, c) for c in self.components)
        return result/normalization

    def moles_(self, species):
        "Number of moles of species or elements."
        species = v.Species(species)
        result = S.Zero
        for i in itertools.chain(self.cations, self.anions):
            if list(species.constituents.keys())[0] in i.constituents:
                result += self.M(self._dbe, i)
        # moles is supposed to compute the moles of a pure element, but with a caveat that pycalphad assumes sum(moles(c) for c in comps) == 1
        # The correct solution is to make the changes where pycalphad assumes n=1. But I think it would be easier to change how we implement the model so that the model has n=1 and the energies are normalized to per-mole-atoms.
        # Since normalizing to moles of quadruplets is allowing us to easily compare with thermochimica, I'm thinking that we might be able to fake pycalphad into thinking we have N=1 by normalizing "moles" to n=1
        # The energies will not be normalized to moles of atoms (and so you cannot yet use this Model to compare to other phases), but internally it should be correct and in agreement with thermochimica
        return result

    @property
    def normalization(self):
        """Divide by this normalization factor to convert from J/mole-quadruplets to J/mole-atoms"""
        return sum(self.M(self._dbe, c) for c in self.components)


    def get_multiphase_constraints(self, conds):
        fixed_chempots = [cond for cond in conds.keys() if isinstance(cond, v.ChemicalPotential)]
        multiphase_constraints = []
        for statevar in sorted(conds.keys(), key=str):
            if not is_multiphase_constraint(statevar):
                continue
            if isinstance(statevar, v.MoleFraction):
                multiphase_constraints.append(Symbol('NP') * self.moles(statevar.species))
            elif statevar == v.N:
                multiphase_constraints.append(Symbol('NP') * (sum(self.moles(spec) for spec in self.nonvacant_elements)))
            elif statevar in [v.T, v.P]:
                return multiphase_constraints.append(S.Zero)
            else:
                raise NotImplementedError
        return multiphase_constraints


    def traditional_reference_energy(self,dbe):
        Gibbs={}
        pair_query = (
            (where('phase_name') == self.phase_name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._pair_test))
        )
        cations = self.cations
        anions = self.anions
        params = dbe._parameters.search(pair_query)
        p = self._p
        surf=S.Zero
        for param in params:
            subl_1 = param['constituent_array'][0]
            subl_2 = param['constituent_array'][1]
            A=subl_1[0]
            B=subl_1[0]
            X=subl_2[0]
            Y=subl_2[0]
            Gibbs[A,B,X,Y]=param['parameter']

        for i, A in enumerate(cations):
            for B in cations[i:]:
                for j, X in enumerate(anions):
                    for Y in anions[j:]:
                        term1=((abs(X.charge)/self.Z(dbe,X,A,B,X,Y))+(abs(Y.charge)/self.Z(dbe,Y,A,B,X,Y)))**(-1)
                        term2=(abs(X.charge)*self.Z(dbe,A,A,A,X,X)/(2*self.Z(dbe,A,A,B,X,Y)*self.Z(dbe,X,A,B,X,Y)))*(Gibbs[A,A,X,X]*2/self.Z(dbe,A,A,A,X,X))
                        term3=(abs(X.charge)*self.Z(dbe,B,B,B,X,X)/(2*self.Z(dbe,B,A,B,X,Y)*self.Z(dbe,X,A,B,X,Y)))*(Gibbs[B,B,X,X]*2/self.Z(dbe,B,B,B,X,X))
                        term4=(abs(Y.charge)*self.Z(dbe,A,A,A,Y,Y)/(2*self.Z(dbe,A,A,B,X,Y)*self.Z(dbe,Y,A,B,X,Y)))*(Gibbs[A,A,Y,Y]*2/self.Z(dbe,A,A,A,Y,Y))
                        term5=(abs(Y.charge)*self.Z(dbe,B,B,B,Y,Y)/(2*self.Z(dbe,B,A,B,X,Y)*self.Z(dbe,Y,A,B,X,Y)))*(Gibbs[B,B,Y,Y]*2/self.Z(dbe,B,B,B,Y,Y))
                        final_term=term1*(term2+term3+term4+term5)
                        surf+=p(A,B,X,Y)*final_term
        return surf/self.normalization


    def reference_energy(self, dbe):
        """
        Returns the weighted average of the endmember energies
        in symbolic form.
        """
        pair_query = (
            (where('phase_name') == self.phase_name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._pair_test))
        )
        self._ξ = S.Zero
        params = dbe._parameters.search(pair_query)
        terms = S.Zero
        for param in params:
            A = param['constituent_array'][0][0]
            X = param['constituent_array'][1][0]
            ξ_AX = self.ξ(A, X)
            self._ξ += ξ_AX
            G_AX = param['parameter']
            Z = self.Z(dbe, A, A, A, X, X)
            terms += (ξ_AX * G_AX)*2/Z
#        print(terms,self.normalization)
        return terms#/self.normalization

    def ideal_mixing_energy(self, dbe):
        # notational niceties
        M = partial(self.M, dbe)
        ϑ = partial(self.ϑ, dbe)
        ξ = self.ξ
        w = self.w
        p = self._p

        cations = self.cations
        anions = self.anions

        Sid = S.Zero
        self.t1 = S.Zero
        self.t2 = S.Zero
        self.t3 = S.Zero
        self.t4 = S.Zero
        ζ = 2.4  # hardcoded, but we can get it from the model_hints (SUBQ) or the pairs (SUBG)
        for A in cations:
            Sid += M(A)*log(ϑ(A))  # term 1
            self.t1 += M(A)*log(ϑ(A))
        for X in anions:
            Sid += M(X)*log(ϑ(X))  # term 2
            self.t2 += M(X)*log(ϑ(X))
        for A in cations:
            for X in anions:
                ξ_AX = ξ(A,X)
                p_AAXX = p(A,A,X,X)
                w_A = w(A)
                w_X = w(X)
                Sid += 4/ζ*ξ_AX*log(ξ_AX/(w_A*w_X))  # term 3
                self.t3 += 4/ζ*ξ_AX*log(ξ_AX/(w_A*w_X))
        # flatter loop over all quadruplets:
        # for A, B, X, Y in ((A, B, X, Y) for i, A in enumerate(cations) for B in cations[i:] for j, X in enumerate(anions) for Y in anions[j:]):
        # Count last 4 terms in the sum
        for i, A in enumerate(cations):
            for B in cations[i:]:
                for j, X in enumerate(anions):
                    for Y in anions[j:]:
                        factor = 1
                        if A != B: factor *= 2
                        if X != Y: factor *= 2
                        Sid += p(A,B,X,Y)*log(p(A,B,X,Y)/(factor * (ξ(A,X)**(1))*(ξ(A,Y)**(1))*(ξ(B,X)**(1))*(ξ(B,Y)**(1)) / ((w(A)**(1))*(w(B)**(1))*(w(X)**(1))*(w(Y)**(1)))))
                        self.t4 += p(A,B,X,Y)*log(p(A,B,X,Y)/(factor * ξ(A,X)*ξ(A,Y)*ξ(B,X)*ξ(B,Y) / (w(A)*w(B)*w(X)*w(Y))))
        return Sid*v.T*v.R/self.normalization


    def excess_mixing_t1(self,dbe,constituent_array):
#        Exid=S.Zero
        Z = partial(self.Z, dbe)
        cations=self.cations
        anions=self.anions
        p = self._p

        subl_1 = constituent_array[0]
        subl_2 = constituent_array[1]
        A=subl_1[0]
        B=subl_1[1]
        X=subl_2[0]
        Y=subl_2[1]
##Figure out how to connect this. Below is the correct expression. Maybe this can be its own function separately
#And it can be called in the other final function
        return 0.5*(p(A,B,X,Y)
                    +sum(0.5*Z(X,A,B,X,Y)*sum(p(A,B,X,Y)/Z(X,A,B,X,Y) for i in anions if i==X!=Y) for j in anions if j==X==Y)
                    +sum(0.5*Z(A,A,B,X,Y)*sum(p(A,B,X,Y)/Z(A,A,B,X,Y) for r in cations if r==A!=B) for q in cations if q==A==B))


    def X_1_2(self,dbe, constituent_array, diffusing_species):
        chem_groups_cat=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['cations']
        chem_groups_an=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['anions']

        cations=self.cations
        anions=self.anions
        p = self._p
        res1=S.Zero
        res2=S.Zero

        subl_1 = constituent_array[0]
        subl_2 = constituent_array[1]
        A=subl_1[0]
        B=subl_1[1]
        X=subl_2[0]
        Y=subl_2[1]
        non_diff_spe=[i for i in subl_1 if i !=diffusing_species][0]

        k_As_cat=[i for i in cations if chem_groups_cat[i]!=chem_groups_cat[diffusing_species] and i not in subl_1]
        l_As_cat=[i for i in cations if chem_groups_cat[i]!=chem_groups_cat[non_diff_spe] and not i in subl_1]

####This is all assuming that there will be only two groups for symmetrical and asymmetrical
        if X==Y and diffusing_species in subl_1:
            As_diff=[diffusing_species]
            if Counter(k_As_cat)!=Counter(l_As_cat):
                As_diff.extend(l_As_cat)
            for count,a in enumerate(As_diff):
                for b in As_diff[count:]:
                    res1+=p(a,b,X,Y)
            res1+=0.5*sum(p(A,B,X,Y) for Y in anions if Y!=X)

            if Counter(k_As_cat)!=Counter(l_As_cat):
                subl_1=list(subl_1)
                subl_1.extend(k_As_cat)
                subl_1.extend(l_As_cat)

            for count,a in enumerate(subl_1):
                for b in subl_1[count:]:
                    res2+=p(a,b,X,Y)+0.5*sum(p(a,b,X,Y) for Y in anions if Y!=X)
        return res1/res2

    def id_symm(self,dbe,A,B,C):
        cations=self.cations
        anions=self.anions
        chem_groups_cat=dbe.phases[phase_name].model_hints['mqmqa']['chemical_groups']['cations']
        chem_groups_an=dbe.phases[phase_name].model_hints['mqmqa']['chemical_groups']['anions']
        in_lis=[A,B,C]
        if set(in_lis).issubset(cations) is True:
            chem_lis=[i for i in in_lis if chem_groups_cat[i]!=chem_groups_cat[A]]
        if set(in_lis).issubset(anions) is True:
            chem_lis=[i for i in in_lis if chem_groups_an[i]!=chem_groups_an[A]]
        if len(chem_lis)==1:
            symm_check=chem_lis[0]
        elif len(chem_lis)>1:
            symm_check_2=set(in_lis)-set(chem_lis)
            symm_check=symm_check_2[0]
        else:
            symm_check=0
        return symm_check


    def K_1_2(self,dbe,A,B):
        w = self.w
        num_res=S.Zero
        chem_groups_cat=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['cations']
        chem_groups_an=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['anions']
        cations=self.cations
        anions=self.anions
        result_cons=[A,B]
        the_rest=set(cations)-set(result_cons)
        result_2=[i for i in the_rest if chem_groups_cat[i]==chem_groups_cat[result_cons[0]] and chem_groups_cat[i]!=chem_groups_cat[result_cons[1]]]
        num_res+=w(A)

        for i in result_2:
            num_res+=w(i)

        return num_res

    def excess_mixing_energy(self,dbe):
        w = self.w
        ξ = self.ξ
        cations=self.cations
        anions=self.anions
        X_a_Xb_tern=1
        pair_query_1 = dbe._parameters.search(
            (where('phase_name') == self.phase_name) & \
            (where('parameter_type') == "EXMG") & \
            (where('constituent_array').test(self._mixing_test))
        )
        pair_query_3 = dbe._parameters.search(
            (where('phase_name') == self.phase_name) & \
            (where('parameter_type') == "EXMQ") & \
            (where('parameter') == 1) & \
            (where('constituent_array').test(self._mixing_test))
        )

        indi_que_3=[i['parameter_order'] for i in pair_query_3]
        chem_groups=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']
        X_ex=S.Zero
        X_tern=S.Zero
        for param in pair_query_1:
            index=param['parameter_order']
            coeff=param['parameter']
            diff=[i for i in indi_que_3 if 0<(i-index)<=4]

            if len(diff)==0:
                X_ex+=self.excess_mixing_t1(dbe,param['constituent_array'])*coeff

            for parse in pair_query_3:
                exp=parse['parameter']
                diff_spe=parse['diffusing_species']
                cons_arr=parse['constituent_array']
                cons_cat=cons_arr[0]
                cons_an=cons_arr[1]
                A=cons_cat[0]
                B=cons_cat[1]
                X=cons_an[0]
                Y=cons_an[1]

                if 0<(parse['parameter_order']-index)<=2 and diff_spe in cons_cat:
                    X_ex_1=coeff*(self.X_1_2(dbe,cons_arr,diff_spe))**exp
                    X_ex+=self.excess_mixing_t1(dbe,cons_arr)*X_ex_1

                elif diff_spe in cations and diff_spe not in cons_cat and \
                0<(parse['parameter_order']-index)<=2:
                    X_tern_diff_spe=parse['parameter_order']-index
                    X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_cat[X_tern_diff_spe])**exp
####Maybe make X_1_2 its own functions and make a different one for X12 ternary
###Or maybe switch it up here?

                elif diff_spe in cations and diff_spe not in cons_cat \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==diff_spe:
                    X_ex+=self.excess_mixing_t1(dbe,cons_arr)\
                    *X_a_Xb_tern*coeff*(ξ(diff_spe,X)/w(X))\
                    *(1-self.K_1_2(dbe,A,B)-self.K_1_2(dbe,B,A))**(exp-1)

                elif diff_spe in cations and diff_spe not in cons_cat \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==0:
                    X_ex+=self.excess_mixing_t1(dbe,cons_arr)\
                    *X_a_Xb_tern*coeff*(ξ(diff_spe,X)/w(X))\
                    *(1-self.K_1_2(dbe,A,B)-self.K_1_2(dbe,B,A))**(exp-1)


                elif diff_spe in cations and diff_spe not in cons_cat \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==B:
                    X_ex+=self.excess_mixing_t1(dbe,cons_arr)\
                    *X_a_Xb_tern*coeff*(ξ(diff_spe,X)/(w(X)*self.K_1_2(dbe,A,B)))\
                    *(1-(ξ(A,X)/(w(X)*self.K_1_2(dbe,A,B))))**(exp-1)




                elif diff_spe in anions and diff_spe not in cons_an \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==diff_spe:
                    X_ex+=self.excess_mixing_t1(dbe,cons_arr)\
                    *X_a_Xb_tern*coeff*(ξ(diff_spe,X)/w(X))\
                    *(1-self.K_1_2(dbe,A,B)-self.K_1_2(dbe,B,A))**(exp-1)

                elif diff_spe in anions and diff_spe not in cons_an \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==0:
                    X_ex+=self.excess_mixing_t1(dbe,cons_arr)\
                    *X_a_Xb_tern*coeff*(ξ(diff_spe,X)/w(X))\
                    *(1-self.K_1_2(dbe,A,B)-self.K_1_2(dbe,B,A))**(exp-1)


                elif diff_spe in anions and diff_spe not in cons_an \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==X:
                    X_ex+=self.excess_mixing_t1(dbe,cons_arr)\
                    *X_a_Xb_tern*coeff*(ξ(diff_spe,X)/(w(X)*self.K_1_2(dbe,A,B)))\
                    *(1-(ξ(A,X)/(w(X)*self.K_1_2(dbe,A,B))))**(exp-1)

        return X_ex/self.normalization
