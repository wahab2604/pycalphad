"""
This module manages interactions with Theano.
"""

import theano
from theano import tensor as tt
import sympy
import sympy.printing.theanocode
from itertools import chain

# Hot patch required for sympy<0.7.7
mapping = sympy.printing.theanocode.mapping
mapping[sympy.And] = tt.and_
mapping[sympy.Or] = tt.or_

global_cache = {}


class TheanoPrinter(sympy.printing.theanocode.TheanoPrinter):
    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop('cache', dict())
        self.dims = kwargs.pop('dims', None)
        super(TheanoPrinter, self).__init__(*args, **kwargs)
    def _print_Symbol(self, s, **kwargs):
        key = s.name
        if key in self.cache:
            print('Cache hit: ', key)
            print('Current cache: ', self.cache)
            return tt.alloc(*chain([self.cache[key]], self.dims))
        else:
            print('Cache miss: ', key)
            print('Current cache: ', self.cache)
            value = tt.tensor(name=s.name, dtype='floatX', broadcastable=[False]*len(self.dims))
            self.cache[key] = value
            return tt.alloc(*chain([value], self.dims))

    def _print_Number(self, n, **kwargs):
        return float(n)


def theano_code(expr, cache=global_cache, **kwargs):
    return TheanoPrinter(cache=cache, settings={}, dims=kwargs.pop('dims')).doprint(expr, **kwargs)