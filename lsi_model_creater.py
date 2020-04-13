import os, json, pickle, keras
import pandas as pd
from utility import *
from lingv_functions import *

#загрузка файлов с данными:
data_rout = r'./data'
models_rout = r'./models'


stopwords_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'stopwords.csv'))
lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'lingv_rules.csv'))
ngrams_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'ngrams.csv'))
texts_collection_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'texts_collection.tsv'), sep = '\t')
rl_ans_df = pd.read_csv(os.path.join(data_rout, 'bss_data', "rules_answers.csv"))
test_acc_df = pd.read_csv(os.path.join(data_rout, 'bss_data', "test_accuracy.csv"))

#подготовка списка синонимов:
sinonims_files = ['01_sinonims.csv', '02_sinonims.csv']
sinonims = []
for file_name in sinonims_files:
    sin_df = pd.read_csv(os.path.join(data_rout, 'bss_data', file_name))
    sinonims.append(list(zip(sin_df["words"], sin_df["initial_forms"])))

ngrams = [list(zip(list(ngrams_df["ngrams"]), list(ngrams_df["token"])))]

print(sinonims[0][:10])
print(ngrams[0][:10])

# соберем LSI модель на основании коллекции из 32 тысяч вопросов:
txt_df = pd.read_csv(os.path.join(data_rout, "bss_data","texts_collection.tsv"), sep = "\t")

# произведем токенизацию, используя имеющуюся модель по БСС (простой токенизатор):
"""
with open(os.path.join(models_rout, "fast_answrs","include_and_model.pickle"), "br") as f:
    model = pickle.load(f)

smp_tkz = SimpleTokenizer(Loader(model))
tknz_txts = smp_tkz.texts_processing(list(txt_df["texts"]))
print(tknz_txts[:10])

lsi_model_dict = lsi_model_create(tknz_txts, topics = 1000)

with open(os.path.join(models_rout, "fast_answrs", "bss_lsi_model_parameters.pickle"), "bw") as f:
    pickle.dump(lsi_model_dict, f)

for i in lsi_model_dict:
    print(i)
"""

with open(os.path.join(models_rout, "lingv_models", "bss_lsi_model_parameters.pickle"), "br") as f:
    lsi_dict = pickle.load(f)


model = {"model_name": "LSI_SIM_model",
        "model_type" : "lsi",
        "etalons" : {
        "texts" : list(lingv_rules_df["words"]), 
        "coeff" : list(lingv_rules_df["coeff"]), 
        "tags" : list(lingv_rules_df["tag"])},            
        "lingvo": [{"sinonims" :  sinonims, "tokenize" : True}, 
        {"ngrams" : ngrams, "tokenize" : False},                
        {"stopwords" : [list(stopwords_df['words'])], "tokenize" : True},
        {"workwords" : [[]], "tokenize" : True}],
        "rules_algorithms" : {"lsi": lsi_dict},
        "texts_algorithms" : {},
        "tokenizer" : "SimpleTokenizer"}
 
with open (os.path.join(models_rout, 'fast_answrs', "lsi_model.pickle"), "bw") as f:
    pickle.dump(model, f)


dct = {"lsi": lsi_dict}
for i in dct["lsi"]:
    print(i)

