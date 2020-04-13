# обучение моделей (NN, LSI, LDA и т. п.) и сохранение обученных моделей

from texts_processors import *
from gensim import similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import Similarity


# Функция 1: формирование tf-idf модели
# загрузим данные для модели tf-idf:
# словарь и корпус для модели формируется на основании файла правил (из правил с признаком TF_IDF_SIM)
# данные: корпус текстов (фрагментов) tknz_texts: ['fragment one', ...] (тексты для создания корпуса)
def tf_idf_model_create(tknz_texts):
    dct = Dictionary(tknz_texts)
    corpus = [dct.doc2bow(tx) for tx in tknz_texts]
    tfidf_model = TfidfModel(corpus)
    return dct, tfidf_model, corpus

# Функция 2: формирование tf модели
# загрузим данные для модели tf:
# словарь и корпус для модели формируется на основании файла правил (из правил с признаком TF_SIM)
def tf_model_create(self):
    rules_corpus_texts = [] #тексты для создания корпуса
    for rule_num in self.logrules:
        for rule, txt, koeff in self.logrules[rule_num]:
            if rule == "TF_SIM":
                rules_corpus_texts.append((rule_num, txt))
    try:
        dct = Dictionary([tx.split() for rule_num, tx in rules_corpus_texts])
        #построим корпус, состоящий из векторов, соответствующих каждому правилу:
        rules_corpus = [(rule_num ,dct.doc2bow(tx.split())) for rule_num, tx in rules_corpus_texts]
        rules, corpus =  zip(*rules_corpus)
        return {"tf_dictionary" : dct, "tf_rules_corpus" : rules_corpus}
    except:            
        return {"tf_dictionary" : None, "tf_rules_corpus" : None}

# Функция 3: формирование LSI модели
def lsi_model_create():
    pass

# Функция 4: формирование частотного словаря для коллекции:
def frequency_dictionary_create():
    pass

# Функция 5: создание словаря биграмм:
def bigrams_dictionary_create():
    pass

if __name__ == "__main__":
    import os, pickle
    import pandas as pd

    data_rout = r"./data"
    txt_df = pd.read_csv(os.path.join(data_rout, "texts_collection.tsv"), sep = "\t")
    print(txt_df)

    models_rout = r"./models"
    with open(os.path.join(models_rout, "include_and_model.pickle"), "br") as f:
        model = pickle.load(f)


    smp_tkz = SimpleTokenizer(Loader(model))
    tknz_txts = smp_tkz.txts_processing(list(txt_df["texts"]))
    print(tknz_txts[:10])
    print(len(tknz_txts))