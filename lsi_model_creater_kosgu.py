import os, pickle
import pandas as pd
from texts_processors import TokenizerApply
from utility import Loader
from lingv_functions import lsi_model_create


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


with open(os.path.join(models_rout, "simplest_model.pickle"), "br") as f:
    model = pickle.load(f)

tknz_txts = TokenizerApply(Loader(model))
# tx = "вчера нам пожелали доброго вечера 345 раз"

tz_txs = tknz_txts.texts_processing(list(train_df["words"]))
print(tz_txs[:10])
print(len(tz_txs))

# подготовка списка синонимов:
stopwords_df = pd.read_csv(os.path.join(data_rout, 'kosgu_data', 'stopwords.csv'))
lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'kosgu_data', 'lingv_rules.csv'))
ngrams_df = pd.read_csv(os.path.join(data_rout, 'kosgu_data', 'ngrams.csv'))


sinonims_files = ['01_sinonims.csv', '02_sinonims.csv']
synonyms = []
for file_name in sinonims_files:
    sin_df = pd.read_csv(os.path.join(data_rout, 'bss_data', file_name))
    synonyms.append(list(zip(sin_df["words"], sin_df["initial_forms"])))
print(synonyms[0][:10])


ngrams = [[(" ".join([w1, w2]), tk) for w1, w2, tk in zip(list(ngrams_df["w1"]), list(ngrams_df["w2"]), list(ngrams_df["bigrams"]))]]
print(ngrams[0][:10])

# соберем LSI модель на основании коллекции из 32 тысяч вопросов:
lsi_model_dict = lsi_model_create(tz_txs, topics=1500)

with open(os.path.join(models_rout, "lingv_models", "kosgu_lsi_model_parameters.pickle"), "bw") as f:
    pickle.dump(lsi_model_dict, f)

with open(os.path.join(models_rout, "lingv_models", "kosgu_lsi_model_parameters.pickle"), "br") as f:
    lsi_dict = pickle.load(f)


model = {"model_name": "kosgu_lsi_model",
        "model_type": "lsi",
        "etalons": {
        "texts": list(lingv_rules_df["words"]),
        "coeff": [0.85]*len(list(lingv_rules_df["coeff"])),
        "tags": list(lingv_rules_df["tag"])},
        "lingvo": [{"sinonims": synonyms, "tokenize": True},
        {"ngrams": ngrams, "tokenize": False},
        {"stopwords": [list(stopwords_df['words'])], "tokenize": True},
        {"workwords": [[]], "tokenize": True}],
        "classificator_algorithms": {},
        "texts_algorithms": lsi_dict,
        "tokenizer": "LsiTokenizer"}
 
with open(os.path.join(models_rout, 'fast_answrs', "kosgu_lsi_model.pickle"), "bw") as f:
    pickle.dump(model, f)


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