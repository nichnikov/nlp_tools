import os, pickle
from utility import Loader

models_rout = r'./models'
with open(os.path.join(models_rout, "fast_answrs", "kosgu_lsi_model.pickle"), "br") as f:
    model = pickle.load(f)

loader_obj = Loader(model)
print(loader_obj.application_field["coeff"])