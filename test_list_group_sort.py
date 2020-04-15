import os, pickle, time, sys
from itertools import groupby
from classificators2 import *

d = [1, 2, 1, 2, 3, 4, 5, 6, 6, 7, 9, 8, 7, 5, 4, 5, 6, 7, 8, 4, 3, 5]
for x, y in groupby(sorted(d)):
    print(x, list(y))


print('\n')

d = [(1, "abc"), (2, "abc"), (1, "abc"), (2, "abc"), (3, "cdf"), (4, "cdf"), (5, "cdf"), (6, "cht"), (6, "cxxdf"), (7, "cdfaa"), (9, "cdfvv"), (8, "cvbf"), 
        (7, "ccv"), (5, "cnn"), (4, "crtw"), (5, "fghhj"), (6, "rtye"), (7, "wwer"), (8, "asdgf"), (4, "lkd"), (3, "uhv"), (5, "rgnm")]

for x, y in groupby(sorted(d, key=lambda x: x[0]), key=lambda x:x[0]):
    print(x, list(y))

print(sorted(d, key=lambda x: x[0]))

print('\n')

print([(x, list(y)) for x, y in groupby(sorted(d, key=lambda x: x[0]), key=lambda x:x[0])])

models_rout = r'./models'

with open(os.path.join(models_rout, "fast_answrs", "kosgu_incl_and_test_model.pickle"), "br") as f:
    model = pickle.load(f)

mdcls = SimpleRules(Loader(model))

model_params = list(zip(mdcls.tknz_model.application_field["tags"], mdcls.tknz_model.application_field["rules"], mdcls.tknz_model.application_field["texts"], mdcls.tknz_model.application_field["coeff"]))

model_params_grouped = [(x, list(y)) for x, y in groupby(sorted(model_params, key=lambda x: x[0]), key=lambda x : x[0])]
#print(model_params_grouped)

for group, rules_list in model_params_grouped:
    for tag, rule, tx, coeff in rules_list:
        print(tag, rule, tx, coeff)
        
        #for tag, rule, tx, coeff in rule_tpl:
        #    print(tag, rule, tx, coeff)
        #print(rule[0], rule[1], rule[2], rule[3])