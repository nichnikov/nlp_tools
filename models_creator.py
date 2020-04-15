import os, json, pickle, keras
import pandas as pd
from utility import *


#загрузка файлов с данными:
data_rout = r'./data'
models_rout = r'./models'

#=============================================================== BSS Models
stopwords_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'stopwords.csv'))
lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'lingv_rules.csv'))
ngrams_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'ngrams.csv'))
#texts_collection_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'texts_collection.tsv'), sep = '\t')
#rl_ans_df = pd.read_csv(os.path.join(data_rout, 'bss_data', "rules_answers.csv"))
#test_acc_df = pd.read_csv(os.path.join(data_rout, 'bss_data', "test_accuracy.csv"))

#подготовка списка синонимов:
sinonims_files = ['01_sinonims.csv', '02_sinonims.csv']
sinonims = []
for file_name in sinonims_files:
    sin_df = pd.read_csv(os.path.join(data_rout, 'bss_data', file_name))
    sinonims.append(list(zip(sin_df["words"], sin_df["initial_forms"])))

ngrams = [list(zip(list(ngrams_df["ngrams"]), list(ngrams_df["token"])))]

model = {"model_name": "bss_include_and_model",
        "model_type" : "simple_rules",
        "etalons" : {
        "rules" : ["include_and"]*len(list(lingv_rules_df["words"])),
        "texts" : list(lingv_rules_df["words"]), 
        "tags" : list(lingv_rules_df["tag"]),
        "coeff" : [0.0]*len(list(lingv_rules_df["tag"]))},            
        "lingvo": [{"sinonims" :  sinonims, "tokenize" : True}, 
        {"ngrams" : ngrams, "tokenize" : False},                
        {"stopwords" : [list(stopwords_df['words'])], "tokenize" : True},
        {"workwords" : [[]], "tokenize" : True}],
        "classificator_algorithms" : {},
        "texts_algorithms" : {},
        "tokenizer" : "SimpleTokenizer"}

with open (os.path.join(models_rout, 'fast_answrs', "bss_include_and_model.pickle"), "bw") as f:
    pickle.dump(model, f)


model = {"model_name": "bss_intersec_share_model",
        "model_type" : "simple_rules",
        "etalons" : {
        "rules" : ["intersec_share"]*len(list(lingv_rules_df["words"])),
        "texts" : list(lingv_rules_df["words"]), 
        "tags" : list(lingv_rules_df["tag"]),
        "coeff" : [0.7]*len(list(lingv_rules_df["tag"]))},            
        "lingvo": [{"sinonims" :  sinonims, "tokenize" : True}, 
        {"ngrams" : ngrams, "tokenize" : False},                
        {"stopwords" : [list(stopwords_df['words'])], "tokenize" : True},
        {"workwords" : [[]], "tokenize" : True}],
        "classificator_algorithms" : {},
        "texts_algorithms" : {},
        "tokenizer" : "SimpleTokenizer"}

with open (os.path.join(models_rout, 'fast_answrs', "bss_intersec_share_model.pickle"), "bw") as f:
    pickle.dump(model, f)

from keras.models import load_model
from gensim.models.doc2vec import Doc2Vec

d2v_model = Doc2Vec.load(os.path.join(models_rout, 'lingv_models', 'bss_doc2vec_model'))
nn_model = load_model(os.path.join(models_rout, 'lingv_models', 'siamese_model_d2v_lstm.h5'))

model = {"model_name": "bss_siamese_lstm_d2v_bss",
        "model_type" : "siamese_lstm_d2v",
        "etalons" : {
        "texts" : list(lingv_rules_df["words"]), 
        "tags" : list(lingv_rules_df["tag"]),
        "coeff": len(list(lingv_rules_df["tag"]))*[0.3]},            
        "lingvo": [{"sinonims" :  sinonims, "tokenize" : True}, 
        {"ngrams" : ngrams, "tokenize" : False},                
        {"stopwords" : [list(stopwords_df['words'])], "tokenize" : True},
        {"workwords" : [[]], "tokenize" : True}],
        "classificator_algorithms" : {"siamese_lstm_model" : nn_model},
        "texts_algorithms" : {"d2v_model" : d2v_model},
        "tokenizer" : "Doc2VecTokenizer"}

with open (os.path.join(models_rout, 'fast_answrs', "bss_siamese_lstm_d2v_bss.pickle"), "bw") as f:
    pickle.dump(model, f)


#======================================= модель для налоговой

lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'tax_dem_data', 'lingv_rules.csv'))
sinonims_df = pd.read_csv(os.path.join(data_rout, 'tax_dem_data', 'sinonims.csv'))

sinonims = list(zip(list(sinonims_df["words"]), list(sinonims_df["initial_forms"])))
model = {"model_name": "tax_demands_simple_model",
        "model_type" : "simple_rules",
        "etalons" : {
        "rules" : list(lingv_rules_df["rules"]),
        "texts" : list(lingv_rules_df["words"]), 
        "tags" : list(lingv_rules_df["tag"]),
        "coeff": list(lingv_rules_df["coeff"])},            
        "lingvo": [{"sinonims" :  sinonims, "tokenize" : True}, 
        {"ngrams" : [[]], "tokenize" : False},                
        {"stopwords" : [[]], "tokenize" : True},
        {"workwords" : [[]], "tokenize" : True}],
        "classificator_algorithms" : None,
        "texts_algorithms" : None,
        "tokenizer" : "SimpleTokenizer"}

with open (os.path.join(models_rout, 'tax_tags', "tax_demands_simple_model.pickle"), "bw") as f:
    pickle.dump(model, f)


#======================================== KOSGU Models

stopwords_df = pd.read_csv(os.path.join(data_rout, 'kosgu_data', 'stopwords.csv'))
lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'kosgu_data', 'lingv_rules.csv'))
ngrams_df = pd.read_csv(os.path.join(data_rout, 'kosgu_data', 'ngrams.csv'))

#подготовка списка синонимов:
sinonims_files = ['01_sinonims.csv', '02_sinonims.csv']
sinonims = []
for file_name in sinonims_files:
    sin_df = pd.read_csv(os.path.join(data_rout, 'kosgu_data', file_name))
    sinonims.append(list(zip(sin_df["words"], sin_df["initial_forms"])))

ngrams = [list(zip(list(ngrams_df["ngrams"]), list(ngrams_df["token"])))]


model = {"model_name": "kosgu_include_and_model",
        "model_type" : "simple_rules",
        "etalons" : {
        "rules" : list(lingv_rules_df["rules"]),
        "texts" : list(lingv_rules_df["words"]), 
        "tags" : list(lingv_rules_df["tag"]),
        "coeff" : list(lingv_rules_df["coeff"])},            
        "lingvo": [{"sinonims" :  sinonims, "tokenize" : True}, 
        {"ngrams" : ngrams, "tokenize" : False},                
        {"stopwords" : [list(stopwords_df['words'])], "tokenize" : True},
        {"workwords" : [[]], "tokenize" : True}],
        "classificator_algorithms" : {},
        "texts_algorithms" : {},
        "tokenizer" : "SimpleTokenizer"}

with open (os.path.join(models_rout, 'fast_answrs', "kosgu_include_and_model.pickle"), "bw") as f:
    pickle.dump(model, f)


#======================================== Labour Protection Models

stopwords_df = pd.read_csv(os.path.join(data_rout, 'labour_protection_data', 'stopwords.csv'))
lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'labour_protection_data', 'lingv_rules.csv'))
ngrams_df = pd.read_csv(os.path.join(data_rout, 'labour_protection_data', 'ngrams.csv'))

#подготовка списка синонимов:
sinonims_files = ['01_sinonims.csv', '02_sinonims.csv']
sinonims = []
for file_name in sinonims_files:
    sin_df = pd.read_csv(os.path.join(data_rout, 'kosgu_data', file_name))
    sinonims.append(list(zip(sin_df["words"], sin_df["initial_forms"])))

ngrams = [list(zip(list(ngrams_df["ngrams"]), list(ngrams_df["token"])))]


model = {"model_name": "labour_protect_include_and_model",
        "model_type" : "simple_rules",
        "etalons" : {
        "rules" : list(lingv_rules_df["rules"]),
        "texts" : list(lingv_rules_df["words"]), 
        "tags" : list(lingv_rules_df["tag"]),
        "coeff" : list(lingv_rules_df["coeff"])},            
        "lingvo": [{"sinonims" :  sinonims, "tokenize" : True}, 
        {"ngrams" : ngrams, "tokenize" : False},                
        {"stopwords" : [list(stopwords_df['words'])], "tokenize" : True},
        {"workwords" : [[]], "tokenize" : True}],
        "classificator_algorithms" : {},
        "texts_algorithms" : {},
        "tokenizer" : "SimpleTokenizer"}

with open (os.path.join(models_rout, 'fast_answrs', "labour_protect_include_and_model.pickle"), "bw") as f:
    pickle.dump(model, f)
