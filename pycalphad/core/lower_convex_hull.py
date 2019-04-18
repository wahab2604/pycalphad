"""
The lower_convex_hull module handles geometric calculations associated with
equilibrium calculation.
"""
from __future__ import print_function
from pycalphad.core.cartesian import cartesian
from pycalphad.core.constants import MIN_SITE_FRACTION
from .hyperplane import hyperplane
import numpy as np
import itertools


def lower_convex_hull(global_grid, state_variables, result_array):
    """
    Find the simplices on the lower convex hull satisfying the specified
    conditions in the result array.

    Parameters
    ----------
    global_grid : Dataset
        A sample of the energy surface of the system.
    state_variables : list
        A list of the state variables (e.g., P, T) used in this calculation.
    result_array : Dataset
        This object will be modified!
        Coordinates correspond to conditions axes.

    Returns
    -------
    None. Results are written to result_array.

    Notes
    -----
    This routine will not check if any simplex is degenerate.
    Degenerate simplices will manifest with duplicate or NaN indices.

    Examples
    --------
    None yet.
    """
    indep_conds = sorted([str(sv) for sv in state_variables])
    comp_conds = sorted([x for x in sorted(result_array.coords.keys()) if x.startswith('X_')])
    comp_conds_indices = sorted([idx for idx, x in enumerate(sorted(result_array.coords['component'].values))
                                 if 'X_'+x in comp_conds])
    comp_conds_indices = np.array(comp_conds_indices, dtype=np.uint64)
    pot_conds = sorted([x for x in sorted(result_array.coords.keys()) if x.startswith('MU_')])
    pot_conds_indices = sorted([idx for idx, x in enumerate(sorted(result_array.coords['component'].values))
                                if 'MU_'+x in pot_conds])
    pot_conds_indices = np.array(pot_conds_indices, dtype=np.uint64)

    if len(set(pot_conds_indices) & set(comp_conds_indices)) > 0:
        raise ValueError('Cannot specify component chemical potential and amount simultaneously')

    if len(comp_conds) > 0:
        cart_values = cartesian([result_array.coords[cond] for cond in comp_conds])
    else:
        cart_values = np.atleast_2d(1.)
    # TODO: Handle W(comp) as well as X(comp) here
    comp_values = np.zeros(cart_values.shape[:-1] + (len(result_array.coords['component'].values),))
    for idx in range(comp_values.shape[-1]):
        if idx in comp_conds_indices:
            comp_values[..., idx] = cart_values[..., np.where(comp_conds_indices == idx)[0][0]]
        elif idx in pot_conds_indices:
            # Composition value not used
            comp_values[..., idx] = 0
        else:
            # Dependent component (composition value not used)
            comp_values[..., idx] = 0
    # Prevent compositions near an edge from going negative
    comp_values[np.nonzero(comp_values < MIN_SITE_FRACTION)] = MIN_SITE_FRACTION*10

    if len(pot_conds) > 0:
        cart_pot_values = cartesian([result_array.coords[cond] for cond in pot_conds])

    specified_conds = set(result_array.coords.keys())
    unspecified_statevars = set(str(x) for x in state_variables) - specified_conds
    free_statevars = {x for x in unspecified_statevars}

    def force_indep_align(da):
        return da.transpose(*itertools.chain(sorted(set(indep_conds) - free_statevars) +
                                             [x for x in da.dims if x not in indep_conds]))
    result_array['GM'] = force_indep_align(result_array.GM)
    result_array['points'] = force_indep_align(result_array.points)
    result_array['MU'] = force_indep_align(result_array.MU)
    result_array['NP'] = force_indep_align(result_array.NP)
    result_array['X'] = force_indep_align(result_array.X)
    result_array['Y'] = force_indep_align(result_array.Y)
    result_array['Phase'] = force_indep_align(result_array.Phase)

    for free_statevar in free_statevars:
        # Use of getitem to avoid xarray.Dataset.T, which is a transpose operation in xarray<0.11
        result_array[free_statevar] = force_indep_align(result_array.__getitem__(free_statevar))
    # factored out via profiling
    result_array_GM_values = result_array.GM.values
    result_array_points_values = result_array.points.values
    result_array_MU_values = result_array.MU.values
    result_array_NP_values = result_array.NP.values
    result_array_X_values = result_array.X.values
    result_array_Y_values = result_array.Y.values
    result_array_Phase_values = result_array.Phase.values
    num_comps = result_array.dims['component']

    it = np.nditer(result_array_GM_values, flags=['multi_index'])
    comp_coord_shape = tuple(len(result_array.coords[cond]) for cond in comp_conds)
    pot_coord_shape = tuple(len(result_array.coords[cond]) for cond in pot_conds)
    while not it.finished:
        grid_idx = []
        for statevar in indep_conds:
            if statevar in free_statevars:
                grid_idx.append(0)
            else:
                grid_idx.append(it.multi_index[result_array.GM.dims.index(statevar)])
        grid_idx = tuple(grid_idx)

        grid = global_grid.isel(**dict(zip(indep_conds, grid_idx)))
        if len(comp_conds) > 0:
            comp_idx = np.ravel_multi_index(tuple(idx for idx, key in zip(it.multi_index, result_array.GM.dims) if key in comp_conds), comp_coord_shape)
            idx_comp_values = comp_values[comp_idx, :]
        else:
            idx_comp_values = np.atleast_1d(1.)
        if len(pot_conds) > 0:
            pot_idx = np.ravel_multi_index(tuple(idx for idx, key in zip(it.multi_index, result_array.GM.dims) if key in pot_conds), pot_coord_shape)
            idx_pot_values = np.array(cart_pot_values[pot_idx, :])

        idx_global_grid_X_values = grid.X.values
        idx_global_grid_GM_values = grid.GM.values
        idx_global_grid_Phase_values = grid.Phase.values
        idx_result_array_MU_values = result_array_MU_values[it.multi_index]
        idx_result_array_MU_values[:] = 0
        for idx in range(len(pot_conds_indices)):
            idx_result_array_MU_values[pot_conds_indices[idx]] = idx_pot_values[idx]
        idx_result_array_NP_values = result_array_NP_values[it.multi_index]
        idx_result_array_points_values = result_array_points_values[it.multi_index]

        # This is a hack to make NP conditions work
        # We add one of each phase to the system
        # Rigorously, we should perform a step/map calculation at the default_value to find all the composition sets
        # "Fixing" a miscibility gap will require a more sophisticated approach
        # If user-specified composition sets were supported, this is where that input would be used
        fixed_phases_present = any(cond.startswith('NP_') for cond in result_array.coords.keys())
        if fixed_phases_present:
            fixed_phase_indices = []
            fixed_phase_amounts = []
            fixed_phase_energies = []
            phases = sorted(set(np.unique(idx_global_grid_Phase_values)) - {'', '_FAKE_'})
            for phase_name in phases:
                # Note: Does not yet support NP(Phase#2) type notation
                phase_grid = np.flatnonzero(idx_global_grid_Phase_values == phase_name)
                first, last = phase_grid[0], phase_grid[-1]
                # Choose fixed phase composition as minimum energy value from the grid
                fixed_index = np.argmin(idx_global_grid_GM_values[first:last+1]) + first
                fixed_phase_indices.append(fixed_index)
                if 'NP_' + phase_name in result_array.coords.keys():
                    cond = 'NP_' + phase_name
                    fixed_phase_amounts.append(result_array.coords[cond].values[it.multi_index[result_array.GM.dims.index(cond)]])
                else:
                    fixed_phase_amounts.append(1./len(phases))
                fixed_phase_energies.append(float(idx_global_grid_GM_values[fixed_index]))
            fixed_phase_indices = np.array(fixed_phase_indices, dtype=np.int64)
            fixed_phase_amounts = np.array(fixed_phase_amounts)
            result_array_GM_values[it.multi_index] = np.dot(fixed_phase_amounts, fixed_phase_energies)
            # Copy phase values out
            points = fixed_phase_indices
            idx_result_array_NP_values[:len(fixed_phase_amounts)] = fixed_phase_amounts
        else:
            result_array_GM_values[it.multi_index] = \
                hyperplane(idx_global_grid_X_values, idx_global_grid_GM_values,
                           idx_comp_values, idx_result_array_MU_values, float(grid.N),
                           pot_conds_indices, comp_conds_indices,
                           np.array([], dtype=np.uint64), np.array([]),
                           idx_result_array_NP_values, idx_result_array_points_values)
            # Copy phase values out
            points = idx_result_array_points_values
        result_array_Phase_values[it.multi_index][:len(points)] = global_grid.Phase.values[grid_idx].take(points, axis=0)
        result_array_X_values[it.multi_index][:len(points),:] = global_grid.X.values[grid_idx].take(points, axis=0)
        result_array_Y_values[it.multi_index][:len(points),:] = global_grid.Y.values[grid_idx].take(points, axis=0)
        # Special case: Sometimes fictitious points slip into the result
        if '_FAKE_' in result_array_Phase_values[it.multi_index]:
            new_energy = 0.
            molesum = 0.
            for idx in range(len(result_array_Phase_values[it.multi_index])):
                midx = it.multi_index + (idx,)
                if result_array_Phase_values[midx] == '_FAKE_':
                    result_array_Phase_values[midx] = ''
                    result_array_X_values[midx] = np.nan
                    result_array_Y_values[midx] = np.nan
                    idx_result_array_NP_values[idx] = np.nan
                else:
                    new_energy += idx_result_array_NP_values[idx] * grid.GM.values[np.index_exp[(points[idx],)]]
                    molesum += idx_result_array_NP_values[idx]
            result_array_GM_values[it.multi_index] = new_energy / molesum
        it.iternext()
    del result_array['points']
    return result_array
