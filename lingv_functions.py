# обучение моделей (NN, LSI, LDA и т. п.) и сохранение обученных моделей
import os, re, pickle, operator
import pandas as pd
import numpy as np
from pymystem3 import Mystem
from texts_processors import SimpleTokenizer
from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary
from collections import defaultdict
from utility import flatten_list, sliceArray, Loader


# Функция 1: формирование tf-idf модели
def tf_idf_model_create(tknz_texts):
    dct = Dictionary(tknz_texts)
    corpus = [dct.doc2bow(tx) for tx in tknz_texts]
    tfidf_model = TfidfModel(corpus)
    return {"dictionary": dct, "model": tfidf_model, "corpus": corpus, "num_topics": None}


# Функция 2: формирование tf модели
def tf_model_create(tknz_texts):
    dct = Dictionary(tknz_texts)
    corpus = [dct.doc2bow(tx) for tx in tknz_texts]
    return {"dictionary": dct, "model": None, "corpus": corpus, "num_topics": None}


# Функция 3: формирование LSI модели
def lsi_model_create(tknz_texts, topics=1000):
    dct = Dictionary(tknz_texts)
    # построим корпус, состоящий из векторов, соответствующих каждому правилу:
    corpus = [dct.doc2bow(tx) for tx in tknz_texts]
    lsi_model = LsiModel(corpus, id2word=dct, num_topics=topics)
    return {"dictionary": dct, "model": lsi_model, "corpus": corpus, "num_topics": topics}


# kwargs : dict = словарь, возвращаемый lsi_model_create
def lsi_model_update(new_texts, **kwargs):
    # добавление "новых документов" - важно для правил
    kwargs["dictionary"].add_documents([tx for tx in new_texts])
    # построим корпус, состоящий из векторов, соответствующих каждому правилу:
    corpus_upd = kwargs["corpus"] + [kwargs["dictionary"].doc2bow(tx) for tx in new_texts]
    lsi_model_upd = LsiModel(corpus_upd, id2word=kwargs["dictionary"], num_topics=kwargs["num_topics"])
    return {"dictionary": kwargs["dictionary"], "model": lsi_model_upd, "corpus": corpus_upd, "num_topics": kwargs["num_topics"]}


# Функция 4: формирование частотного словаря для коллекции:
def frequency_dictionary_create(tknz_texts):
    dict_frequency = defaultdict(int)
    tokens = [i for i in flatten_list(tknz_texts)]
    for token in tokens:
        dict_frequency[token] += 1

    # сортировка словаря по значению
    sort_dict = sorted(dict_frequency.items(), key=operator.itemgetter(1), reverse=True)
    dict_frequency_df = pd.DataFrame(sort_dict, columns=['token', 'quantity'])

    dict_frequency_df['freq'] = dict_frequency_df['quantity']/sum(dict_frequency_df['quantity'])
    dict_frequency_df.sort_values('freq', ascending=False)
    return dict_frequency_df


# Функция 5: создание словаря биграмм:
def bigrams_dictionary_create(tknz_texts):    
    m = Mystem()

    # сформируем датафрейм словаря частотности токенов:
    dfr = frequency_dictionary_create(tknz_texts)
    # список для кортежей, состоящих из токенов и их частей речи
    ws_sbj = []
    for tx in tknz_texts:
        temp_ws_sbj = []
        for anlys_dict in m.analyze(" ".join(tx)):
            try:
                sbj = re.sub('[^A-Z]', '', anlys_dict['analysis'][0]['gr'])
                w = anlys_dict['text']
                temp_ws_sbj.append((w, sbj))
            except Exception:
                print("Exception:", w, sbj)
        ws_sbj.append(temp_ws_sbj)    
    
    # оставим только прилагательные и существительные:
    ws_sbj_sa = [[t for t in x if t[1] in ['A', 'S']] for x in ws_sbj]
    
    # удалим пустые, если такие есть:
    ws_sbj_sa = [x for x in ws_sbj_sa if x != []]
    
    # кандидаты на биграммы: AS, SS (возможно нужно перенести в параметры):
    bigrams_candidate = []
    for q_list in ws_sbj_sa:
        bigrams_candidate.append([x for x in sliceArray(q_list, length=2, stride=2) if ''.join([x[0][1], x[1][1]]) in ['AS', 'SS']])
    
    bigrams_candidate = [x for x in bigrams_candidate if x != []]
    bigrams_candidate_sep = [[(''.join([x[0][0], x[1][0]]), x[0][0], x[1][0]) for x in bg] for bg in bigrams_candidate]
    
    # сделаем список биграмм "плоским"
    flatit = flatten_list(bigrams_candidate_sep)
    bigrams_candidate_sep = [x for x in flatit]
    
    # создадим пандас датафрейм из кандидатов в биграммы
    bigrams_candidate_df = pd.DataFrame(bigrams_candidate_sep, columns = ['bigrams', 'w1', 'w2'])
    
    # посчитаем частотность биграмм и их токенов
    # посчитаем частотность биграмм:
    bgms_cand_freq = bigrams_candidate_df[['bigrams', 'w1']].groupby('bigrams', as_index = False).count()
    bgms_cand_freq.rename(columns = {'w1':'quantity'}, inplace = True)
    bgms_cand_freq['freq'] = bgms_cand_freq['quantity']/sum(bgms_cand_freq['quantity'])

    # вернем слова:
    bgms_freq_words = pd.merge(bgms_cand_freq, bigrams_candidate_df, how='left', on='bigrams', copy = False)
    bgms_freq_words.drop_duplicates(inplace = True)

    # объединим частотный словарь токенов и словарь биграмм
    dfr_w1 = dfr.rename(columns={'freq' : 'w1_freq', 'token':'w1'})
    bigrams_est = pd.merge(bgms_freq_words, dfr_w1[['w1', 'w1_freq']], on='w1')

    dfr_w2 = dfr.rename(columns={'freq' : 'w2_freq', 'token':'w2'})
    bigrams_est = pd.merge(bigrams_est, dfr_w2[['w2', 'w2_freq']], on='w2')
    bigrams_est.rename(columns = {'freq' : 'bigrms_freq'}, inplace = True)

    # теперь все готово к оценке вероятности того, насколько данная биграмма похожа на УСС:
    # количество слов корпуса, участвующих в построении биграмм
    # в каждой биграмме 2 слова
    n = 2*sum(bigrams_est['quantity'])

    # оценка взаимной информации для слов, входящих в биграммы:
    bigrams_est['estimate'] = np.log((n*bigrams_est['bigrms_freq']**3)/(bigrams_est['w1_freq']*bigrams_est['w2_freq']))
    bigrams_est_sort_df = bigrams_est.sort_values('estimate', ascending=False)
    return bigrams_est_sort_df


if __name__ == "__main__":
    data_rout = r"./data"
    txt_df = pd.read_csv(os.path.join(data_rout, "bss_data","texts_collection.tsv"), sep = "\t")
    print(txt_df)

    models_rout = r"./models"
    with open(os.path.join(models_rout, "fast_answrs","bss_include_and_model.pickle"), "br") as f:
        model = pickle.load(f)
    
    smp_tkz = SimpleTokenizer(Loader(model))
    tknz_txts = smp_tkz.texts_processing(list(txt_df["texts"][:1000]))
    print(tknz_txts[:10])
    print(len(tknz_txts))

    dct1 = tf_idf_model_create(tknz_txts)
    print(dct1)

    dct2 = tf_model_create(tknz_txts)
    print(dct2)

    dct3 = lsi_model_create(tknz_txts, topics=10)
    print(dct3)

    # проверка векторизации lsi модели:    
    txt_corp = dct3["dictionary"].doc2bow(tknz_txts[5])
    txt_vect = dct3["model"][txt_corp]
    print(txt_vect)
    print(tknz_txts[5])

    # попробуем "апгрейдить модель" и сравнить результаты "до" и "после"
    tx = 'мой дядя самых честных правил'
    tk_tx = smp_tkz.texts_processing([tx])
    print(tk_tx)

    txt_corp = dct3["dictionary"].doc2bow(tk_tx[0])
    txt_vect = dct3["model"][txt_corp]
    print("before upd", txt_vect)

    dct3 = lsi_model_update(tk_tx, **dct3)
    txt_corp = dct3["dictionary"].doc2bow(tk_tx[0])
    txt_vect = dct3["model"][txt_corp]
    print("after upd", txt_vect)

    fr_dct = frequency_dictionary_create(tknz_txts)
    print(fr_dct)

    bg_df = bigrams_dictionary_create(tknz_txts)
    print(bg_df)