import os, pickle
import pandas as pd
from texts_processors import TokenizerApply
from utility import Loader


# загрузка файлов с данными:
questions_rout = r'/home/alexey/big/data/fast_answers'
data_rout = r'./data'
models_rout = r'./models'

quests_df = pd.read_csv(os.path.join(questions_rout, "gf_request.csv"), header=None)
quests50th_df = pd.DataFrame(quests_df[0].sample(50000))
quests50th_df.rename(columns={0: "words"}, inplace=True)
print(quests50th_df[:100])

etalons_df = pd.read_csv(os.path.join(data_rout, "kosgu_data", "lingv_rules.csv"))
print(etalons_df["words"][:100])
print(etalons_df.shape)

train_df = pd.DataFrame(pd.concat([quests50th_df["words"], etalons_df["words"]], axis=0))
print('\n', train_df)
print(train_df.shape)


"""
# создание простейшей модели для токенизации
model = {"model_name": "simplest_model",
        "model_type": "simple_tokenizer",
        "etalons": {
        "texts": [],
        "coeff": [],
        "tags": []},
        "lingvo": [{"synonyms": [[]], "tokenize": False},
        {"ngrams": [[]], "tokenize": False},
        {"stopwords": [[]], "tokenize": False},
        {"workwords": [[]], "tokenize": False}],
        "classificator_algorithms": {},
        "texts_algorithms": {},
        "tokenizer": "SimpleTokenizer"}

with open(os.path.join(models_rout, "simplest_model.pickle"), "bw") as f:
    pickle.dump(model, f)
"""

with open(os.path.join(models_rout, "simplest_model.pickle"), "br") as f:
    model = pickle.load(f)

tzapl = TokenizerApply(Loader(model))
# tx = "вчера нам пожелали доброго вечера 345 раз"

tz_txs = tzapl.texts_processing(train_df["words"])
print(tz_txs[:10])
print(len(tz_txs))

# подготовка списка синонимов:
"""
stopwords_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'stopwords.csv'))
lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'lingv_rules.csv'))
ngrams_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'ngrams.csv'))
texts_collection_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'texts_collection.tsv'), sep = '\t')
rl_ans_df = pd.read_csv(os.path.join(data_rout, 'bss_data', "rules_answers.csv"))
test_acc_df = pd.read_csv(os.path.join(data_rout, 'bss_data', "test_accuracy.csv"))


sinonims_files = ['01_sinonims.csv', '02_sinonims.csv']
synonyms = []
for file_name in sinonims_files:
    sin_df = pd.read_csv(os.path.join(data_rout, 'bss_data', file_name))
    synonyms.append(list(zip(sin_df["words"], sin_df["initial_forms"])))

ngrams = [list(zip(list(ngrams_df["ngrams"]), list(ngrams_df["token"])))]

print(synonyms[0][:10])
print(ngrams[0][:10])

# соберем LSI модель на основании коллекции из 32 тысяч вопросов:
txt_df = pd.read_csv(os.path.join(data_rout, "bss_data","texts_collection.tsv"), sep = "\t")

# произведем токенизацию, используя имеющуюся модель по БСС (простой токенизатор):
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

with open(os.path.join(models_rout, "lingv_models", "bss_lsi_model_parameters.pickle"), "br") as f:
    lsi_dict = pickle.load(f)


model = {"model_name": "LSI_SIM_model",
        "model_type" : "lsi",
        "etalons" : {
        "texts" : list(lingv_rules_df["words"]), 
        "coeff" : list(lingv_rules_df["coeff"]), 
        "tags" : list(lingv_rules_df["tag"])},            
        "lingvo": [{"sinonims" :  synonyms, "tokenize" : True},
        {"ngrams" : ngrams, "tokenize" : False},                
        {"stopwords" : [list(stopwords_df['words'])], "tokenize" : True},
        {"workwords" : [[]], "tokenize" : True}],
        "classificator_algorithms" : {},
        #"texts_algorithms" : {"lsi": lsi_dict},
        "texts_algorithms" : lsi_dict,
        "tokenizer" : "LsiTokenizer"}
 
with open (os.path.join(models_rout, 'fast_answrs', "lsi_model.pickle"), "bw") as f:
    pickle.dump(model, f)


dct = {"lsi": lsi_dict}
for i in dct["lsi"]:
    print(i)
"""
