"""
This module manages interactions with Theano.
"""

import theano
import numpy as np
from theano import tensor as tt
import sympy
from functools import partial
from itertools import chain
from collections import OrderedDict
import sympy.printing.theanocode

# Hot patch required for sympy<0.7.7
mapping = sympy.printing.theanocode.mapping
mapping[sympy.And] = tt.and_
mapping[sympy.Or] = tt.or_

global_cache = {}


class TheanoPrinter(sympy.printing.theanocode.TheanoPrinter):
    """ Code printer for Theano computations with some modifications for broadcasting and stability"""
    printmethod = "_theano_pycalphad"

    def _print_Symbol(self, s, dtypes={}, broadcastables={}, dims=tuple(), alloc=False):
        dtype = dtypes.get(s, 'floatX')
        broadcastable = broadcastables.get(s, ())
        key = (s.name, dtype, broadcastable, type(s), alloc)
        if key in self.cache:
            #print('Cache hit: ', key)
            #print('Current cache: ', self.cache)
            return self.cache[key]
        else:
            #print('Cache miss: ', key)
            #print('Current cache: ', self.cache)
            value = tt.tensor(name=s.name, dtype='floatX', broadcastable=broadcastable)
            self.cache[key] = value
            if alloc:
                self.cache[key] = tt.alloc(*chain([value], dims))
            else:
                self.cache[key] = value
            return self.cache[key]

    def _print_Integer(self, expr, **kwargs):
        # Upcast to work around an integer overflow bug
        return np.asarray(expr.p, dtype=theano.config.floatX)

    def _print_Rational(self, expr, **kwargs):
        # Upcast to work around an integer overflow bug
        pxx = np.asarray(self._print(expr.p, **kwargs), dtype=theano.config.floatX)
        qxx = np.asarray(self._print(expr.q, **kwargs), dtype=theano.config.floatX)
        return tt.true_div(pxx, qxx)

    def _print_Number(self, n, **kwargs):
        return float(n)


def theano_code(expr, cache=global_cache, **kwargs):
    return TheanoPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def _build_grad_hess(obj, flat_params, full_shape, num_vars):
    grad = theano.grad(obj.sum(), flat_params)
    hess, u = theano.scan(lambda i, gp, p: theano.grad(gp[i], p),
                          sequences=tt.mod(tt.arange(flat_params.shape[0]), num_vars * np.multiply.reduce(full_shape)),
                          non_sequences=[grad, flat_params])
    # Not pretty, but it works
    hess = hess.reshape((-1, np.multiply.reduce(full_shape))).sum(axis=-1)
    hess = hess.reshape((num_vars, np.multiply.reduce(full_shape), num_vars)).transpose(1, 0, 2)
    hess = hess.reshape(full_shape + [num_vars, num_vars])

    # Send 'gradient' axis to back
    grad = grad.reshape([num_vars] + full_shape).transpose(*chain(range(len(full_shape)+1)[1:], [0]))
    return grad, hess


def build_functions(sympy_graph, indep_vars, site_fracs, broadcast_dims=None):
    """Convert a sympy graph to Theano functions for f(...), its gradient and its Hessian."""
    ndims = len(indep_vars) + 1  # e.g., T, P, and site fractions
    nvars = len(indep_vars) + len(site_fracs)

    if broadcast_dims is None:
        broadcast_matrix = np.ones((nvars, ndims), dtype=np.bool)
        # Don't broadcast independent variable over its own dimension
        # We assume indep_vars is already in "axis order" for broadcasting purposes
        broadcast_matrix[np.arange(len(indep_vars)), np.arange(len(indep_vars))] = False
        # Site fractions are never broadcast over any dimension; always a "full" ndarray
        broadcast_matrix[len(indep_vars):, :] = False
        broadcast_dims = {s: tuple(broadcast_matrix[i]) for i, s in enumerate(indep_vars + site_fracs)}
    # All this cache nonsense is necessary because we have to make sure our symbolic variables are
    # exactly the same copy as what appears in the graph, or else Theano will say it's disconnected
    cache = OrderedDict()
    for s in indep_vars:
        def_key = (s.name, 'floatX', broadcast_dims[s], type(s), False)
        cache[def_key] = tt.tensor(name=s.name, dtype='floatX', broadcastable=broadcast_dims[s])
    for s in site_fracs:
        def_key = (s.name, 'floatX', broadcast_dims[s], type(s), False)
        cache[def_key] = tt.tensor(name=s.name, dtype='floatX', broadcastable=broadcast_dims[s])
    first_sitefrac_key = (site_fracs[0].name, 'floatX', broadcast_dims[site_fracs[0]],
                          type(site_fracs[0]), False)
    # Dimension array of full broadcast shape, in symbolic form
    dims = [cache[first_sitefrac_key].shape[i] for i in range(len(broadcast_dims[site_fracs[0]]))]

    # flat concatenation to use in Hessian
    flat_params = tt.join(*chain([0], [tt.alloc(*chain([cache[i]], dims)).flatten() for i in cache.keys()]))
    # reconstructed variables to use to compute obj
    flat_vars = tt.split(flat_params, nvars * [np.multiply.reduce(dims)], n_splits=nvars)
    new_vars = {key[0]: flat_vars[idx].reshape(dims) for idx, key in enumerate(cache.keys())}
    for s in indep_vars+site_fracs:
        alloc_key = (s.name, 'floatX', broadcast_dims[s], type(s), True)
        cache[alloc_key] = new_vars[s.name]
    code = partial(theano_code, cache=cache, broadcastables=broadcast_dims, dims=dims, alloc=True)
    tinputs = [cache[(s.name, 'floatX', broadcast_dims[s], type(s), False)] for s in indep_vars + site_fracs]
    obj = list(map(code, [sympy_graph]))
    obj = obj[0] if len(obj) == 1 else obj
    grad, hess = _build_grad_hess(obj, flat_params, dims, nvars)
    custom_opt = theano.compile.optdb.query(theano.gof.Query(include=['fast_compile'], exclude=['inplace']))
    custom_mode = theano.compile.mode.Mode(linker='cvm', optimizer=custom_opt)
    ofunc = theano.function(tinputs, obj, mode=custom_mode)
    gfunc = theano.function(tinputs, grad, mode=custom_mode)
    hfunc = theano.function(tinputs, hess, mode=custom_mode)
    return ofunc, gfunc, hfunc
