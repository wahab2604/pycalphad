""""Common pyparsing grammar patterns."""

from pyparsing import alphas, nums
from pyparsing import Group, OneOrMore, Optional, Regex, Suppress, Word
import re

pos_neg_int_number = Word('+-' + nums).setParseAction(lambda t: [int(t[0])])  # '+3' or '-2' are examples
# matching float w/ regex is ugly but is recommended by pyparsing
regex_after_decimal = r'([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
float_number = Regex(r'[-+]?([0-9]+\.(?!([0-9]|[eE])))|{0}'.format(regex_after_decimal)) \
    .setParseAction(lambda t: [float(t[0])])

chemical_formula = Group(OneOrMore(Word(alphas, min=1, max=2) + Optional(float_number, default=1.0))) + \
                   Optional(Suppress('/') + pos_neg_int_number, default=0)

reg_symbol = r'([A-Za-z][A-Za-z]?)'  # Don't match characters between `Z` and `a` like `[`
reg_amount = r'([-+]?([0-9]+\.(?!([0-9]|[eE])))|([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?))?'
chem_regex = reg_symbol + reg_amount
reg_charge = r'/([+-]?[0-9]+)'  # FE/+2 FE/3 CL/-1
reg_postfix_charge = r'\[([0-9]+)?([-+])\]'  # FE[2+] CU[+] CL[-] O[2-]

def parse_chemical_formula(formula):
    """"""
    matches = re.findall(chem_regex, formula)
    sym_amnts = [(m[0], float(m[1]) if m[1] != '' else 1.0) for m in matches]
    # try both methods of charge, falling back to postfix +/-
    charge = re.search(reg_charge, formula)
    if charge is not None:
        charge = int(charge.groups()[0])
    else:
        charge = re.search(reg_postfix_charge, formula)  # try [charge+/-]
        if charge is not None:
            # 2 groups: (Optional(number), +/-)
            # Number may not exist (i.e. CL[-]) so we get one.
            charge = int(f'{charge.groups()[1]}{charge.groups()[0] or 1}')
        else:
            charge = 0
    return (sym_amnts, charge)
