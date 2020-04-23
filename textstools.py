'''
идея в том, чтобы создать класс, объединяющий в себе методы обработки текста (замену синонимов, замену n-грамм) 
и данные (массив синонимов и n-грамм, набор правил) для такой обработки
Пока не ясно, что должно являться экземпляром класса и на каком 
'''
import os, re, pickle, difflib, operator, time
from pymystem3 import Mystem
import pandas as pd
import numpy as np
from gensim import similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary
from gensim.similarities import Similarity
from collections import defaultdict

""" Утилитарные функции над массивами """
"""Нарезает массив окном размера len c шагом stride"""


def flatten_all(iterable):
    for elem in iterable:
        if not isinstance(elem, list):
            yield elem
        else:
            for x in flatten_all(elem):
                yield x


def sliceArray(src: [], length: int = 1, stride: int = 1):
    return [src[i:i + length] for i in range(0, len(src), stride) if len(src[i:i + length]) == length]


'''Нарезает массив окном размера len с шагом в размер окна'''


def splitArray(src: [], length: int):
    return sliceArray(src, length, length)


"""Преобразует массив токенов в мешок (каждый токен представлен кортежем -- (токен, сколько раз встречается в массиве))"""


def arr2bag(src: []):
    return [(x, src.count(x)) for x in set(src)]


"""Возвращает массив токенов src за исключением rem"""


def removeTokens(src: [], rem: []):
    return [t for t in src if t not in rem]


"""Заменяет в массиве src множество токенов аскриптора asc дексрипторами токена (синонимия)"""


def replaceAscriptor(src: [], asc: [], desc: []):
    src_repl = []
    length = len(asc)
    src_ = [src[i:i + length] for i in range(0, len(src), 1)]
    i = 0
    while i < len(src_):
        if src_[i] == asc:
            src_repl = src_repl + desc
            i += length
        else:
            src_repl.append(src_[i][0])
            i += 1
    return src_repl


def dictionary_frequency(texts_collections):
    dict_frequency = defaultdict(int)
    tokens_iterator = flatten_all([tx.split() for tx in texts_collections])
    tokens = [i for i in tokens_iterator]
    for token in tokens:
        dict_frequency[token] += 1

    # сортировка словаря по значению
    sort_dict = sorted(dict_frequency.items(), key=operator.itemgetter(1), reverse=True)
    dict_frequency_df = pd.DataFrame(sort_dict, columns=['token', 'quantity'])

    dict_frequency_df['freq'] = dict_frequency_df['quantity'] / sum(dict_frequency_df['quantity'])
    dict_frequency_df.sort_values('freq', ascending=False)
    return dict_frequency_df


# класс для предварительной обработки
# лемматизирует
# удаляет стоп-слова
# заменяет синонимы
# переменные: kwargs["stopwords"], kwargs["sinonims"], kwargs["ngrams"]
class TokensHandling():
    def __init__(self, **kwargs):
        self.m = Mystem()
        self.kwargs = kwargs
        self.sinonims = self.sinonims_fill()  # список синонимов и "канонических" форм
        self.ngrams = self.ngrams_fill()  # список n-грамм
        self.stopwords = self.stopwords_fill()

    # функция, проводящая предобработку текста (включая лемматизацию)
    def texts_handl(self, txt: str):
        try:
            lemm_txt = ' '.join(self.m.lemmatize(txt.lower()))
            lemm_txt = re.sub('[^a-zа-я\d]', ' ', lemm_txt)
            lemm_txt = re.sub('\s+', ' ', lemm_txt)
            return lemm_txt
        except:
            return txt

    # функция, замещающуя синонимы в тексте:
    def sinonims_handle(self, text):
        for sin_tuples in self.sinonims:
            text = self.patterns_change_split(sin_tuples, text)
        return text

    # надо поменять на работу с токенами (отказаться от регулярных выражений):
    def patterns_change_split(self, asc_dsc_tuples, text_for_change: str):
        for asc, dsc in asc_dsc_tuples:
            text_for_change = ' '.join(replaceAscriptor(text_for_change.split(), asc.split(), dsc.split()))
        return text_for_change

    def stopwords_fill(self):
        stopwords_df = self.kwargs["stopwords"]
        stopwords_df['lemm_words'] = stopwords_df['words'].apply(self.texts_handl)
        stopwords_df['working_text'] = stopwords_df['lemm_words'].apply(lambda tx: self.sinonims_handle(tx))
        stopwords_df['working_text'] = stopwords_df['working_text'].apply(
            lambda tx: self.patterns_change_split(self.ngrams, tx))
        return list(stopwords_df['working_text'])

    def sinonims_fill(self):
        sinonims_list = []
        for sinonims_df in self.kwargs["sinonims"]:
            sinonims_df['lemm_words'] = sinonims_df['words'].apply(self.texts_handl)
            sinonims_df['lemm_init_forms'] = sinonims_df['initial_forms'].apply(self.texts_handl)
            sinonims_list.append(list(zip(sinonims_df['lemm_words'], sinonims_df['lemm_init_forms'])))
        return sinonims_list

    def ngrams_fill(self):
        ngrams_df = self.kwargs["ngrams"]
        # ngrams_df['lemm_ngrams'] = ngrams_df['ngrams'].apply(self.texts_handl)
        # ngrams_df['lemm_ngrams'] = ngrams_df['lemm_ngrams'].apply(lambda tx: self.sinonims_handle(tx))
        # return list(zip(ngrams_df['lemm_ngrams'],  ngrams_df['token']))
        return list(zip(ngrams_df['ngrams'], ngrams_df['token']))


# класс, который применяет к текстам токенезацию, словари и синонимы
# text_list - ['text1', 'text2', ...] - список текстов для обработки, kwargs['sinonims']
# синонимы, kwargs["ngrams"] - n-граммы, kwargs["stopwords"] - стоп-слова
class TextHandling(TokensHandling):
    def __init__(self, texts_list, **kwargs):
        self.kwargs = kwargs
        self.m = self.kwargs['m']
        self.texts_list = texts_list
        self.sinonims = kwargs['sinonims']
        self.ngrams = kwargs["ngrams"]
        self.stopwords = kwargs["stopwords"]
        self.lemm_texts = self.texts_handling()

    # замена стоп-слов без регулярных выражений, удаление токенов
    def stopwords_dell_split(self, text_for_change: str):
        for stopword in self.stopwords:
            text_for_change = ' '.join(
                [x for x in flatten_all(replaceAscriptor(text_for_change.split(), stopword.split(), ['']))])
        return text_for_change

    # функция, которая возвращает текст, лемматизированный с замененными синонимами и n-граммами, приведенный к токенам из словаря:
    def text_prepare(self, text):
        # (1)лемматизация:
        lemm_tx = self.texts_handl(text)
        # (2)замена синонимов и n-грамм
        lemm_tx = self.sinonims_handle(lemm_tx)
        lemm_tx = self.patterns_change_split(self.ngrams, lemm_tx)  # замена на сплит
        lemm_tx = self.stopwords_dell_split(lemm_tx)
        # вернем лемматизированный текст с замененными синонимами и n-граммами
        return lemm_tx

    def texts_handling(self):
        lemm_texts_list = []
        k = 0
        for txt in self.texts_list:
            print(k)
            k += 1
            try:
                lemm_text = self.text_prepare(txt)
                lemm_texts_list.append(lemm_text)
            except:
                print("TextHandlError:", txt)
        return lemm_texts_list


# класс для извлечения сущностей из коллекции
# построение частотного словаря
# автоматическое извлечение n-грамм
# kwargs["texts_collection"] - список текстов, kwargs["sinonims"] - синонимы, kwargs["ngrams"] - n-граммы
# ЗАМЕЧАНИЕ: (?) нужно добавить удаление стоп-слов ??? (или не нужно)
class CollectionStatHandling(TokensHandling):
    def __init__(self, texts_collection, **kwargs):
        self.kwargs = kwargs
        self.texts_collection = texts_collection
        self.m = self.kwargs["m"]
        self.sinonims = kwargs["sinonims"]
        self.ngrams = kwargs["ngrams"]
        self.dict_frequency_df = self.dictionary_frequency()  # частотный словарь [(token, freq), (), ...] - список кортежей, из токена и его частоты в коллекции
        self.bigrams_df = self.bigrams_estimate()  #
        self.gensim_dictionary, self.gensim_corpus = self.dictionary_corp_fill()  # словарь нумерованных токенов (объект gensim)

    # сформируем словарь и корпус на текстовой коллекции, которая позволит строить более качественные статистические модели
    def dictionary_corp_fill(self):
        data_df = pd.DataFrame(self.texts_collection, columns=['texts'])
        data_df['lemm_texts'] = data_df['texts'].apply(self.texts_handl)
        data_df['working_text'] = data_df['lemm_texts'].apply(lambda tx: self.sinonims_handle(tx))
        data_df['working_text'] = data_df['working_text'].apply(lambda tx: self.patterns_change_split(self.ngrams, tx))
        texts_collections = [x for x in list(data_df['working_text'])]
        dict_df = self.dict_frequency_df[self.dict_frequency_df[
                                             'freq'] >= 0.0005]  # частота токенов, лимитирующая количество, выбрана волюнтаристски
        dictionary = Dictionary([list(dict_df['token'])])
        corpus = [dictionary.doc2bow(tx.split()) for tx in texts_collections]
        return dictionary, corpus

    # функция, определяющая (добавляющая к леммам) части речи:
    # def lemm_texts_with_parts_of_speech(self):
    def bigrams_estimate(self):
        ws_sbj = []  # список для кортежей, состоящих из токенов и их частей речи
        for tx in self.texts_collection:
            temp_ws_sbj = []
            for anlys_dict in self.m.analyze(tx):
                try:
                    sbj = re.sub('[^A-Z]', '', anlys_dict['analysis'][0]['gr'])
                    w = anlys_dict['text']
                    temp_ws_sbj.append((w, sbj))
                except Exception:
                    None
            ws_sbj.append(temp_ws_sbj)

            # оставим только прилагательные и существительные:
        ws_sbj_sa = [[t for t in x if t[1] in ['A', 'S']] for x in ws_sbj]

        # удалим пустые, если такие есть:
        ws_sbj_sa = [x for x in ws_sbj_sa if x != []]

        # кандидаты на биграммы: AS, SS (возможно нужно перенести в параметры):
        bigrams_candidate = []
        for q_list in ws_sbj_sa:
            bigrams_candidate.append(
                [x for x in sliceArray(q_list, length=2, stride=2) if ''.join([x[0][1], x[1][1]]) in ['AS', 'SS']])

        bigrams_candidate = [x for x in bigrams_candidate if x != []]
        bigrams_candidate_sep = [[(''.join([x[0][0], x[1][0]]), x[0][0], x[1][0]) for x in bg] for bg in
                                 bigrams_candidate]

        # сделаем список биграмм "плоским"
        flatit = flatten_all(bigrams_candidate_sep)
        bigrams_candidate_sep = [x for x in flatit]

        # создадим пандас датафрейм из кандидатов в биграммы
        bigrams_candidate_df = pd.DataFrame(bigrams_candidate_sep, columns=['bigrams', 'w1', 'w2'])

        # посчитаем частотность биграмм и их токенов
        # посчитаем частотность биграмм:
        bgms_cand_freq = bigrams_candidate_df[['bigrams', 'w1']].groupby('bigrams', as_index=False).count()
        bgms_cand_freq.rename(columns={'w1': 'quantity'}, inplace=True)
        bgms_cand_freq['freq'] = bgms_cand_freq['quantity'] / sum(bgms_cand_freq['quantity'])

        # вернем слова:
        bgms_freq_words = pd.merge(bgms_cand_freq, bigrams_candidate_df, how='left', on='bigrams', copy=False)
        bgms_freq_words.drop_duplicates(inplace=True)

        # переведем словарь частотности токенов в пандас датафрейм:
        dfr = self.dict_frequency_df

        # объединим частотный словарь токенов и словарь биграмм
        dfr_w1 = dfr.rename(columns={'freq': 'w1_freq', 'token': 'w1'})
        bigrams_est = pd.merge(bgms_freq_words, dfr_w1[['w1', 'w1_freq']], on='w1')

        dfr_w2 = dfr.rename(columns={'freq': 'w2_freq', 'token': 'w2'})
        bigrams_est = pd.merge(bigrams_est, dfr_w2[['w2', 'w2_freq']], on='w2')
        bigrams_est.rename(columns={'freq': 'bigrms_freq'}, inplace=True)

        # теперь все готово к оценке вероятности того, насколько данная биграмма похожа на УСС:
        # количество слов корпуса, участвующих в построении биграмм
        N = 2 * sum(bigrams_est['quantity'])  # в каждой биграмме 2 слова

        # оценка взаимной информации для слов, входящих в биграммы:
        bigrams_est['estimate'] = np.log(
            (N * bigrams_est['bigrms_freq'] ** 3) / (bigrams_est['w1_freq'] * bigrams_est['w2_freq']))
        bigrams_est_sort_df = bigrams_est.sort_values('estimate', ascending=False)
        return bigrams_est_sort_df

    # формирование частотного словаря
    def dictionary_frequency(self):
        dict_frequency = defaultdict(int)
        tokens_iterator = flatten_all([tx.split() for tx in self.texts_collection])
        tokens = [i for i in tokens_iterator]
        for token in tokens:
            dict_frequency[token] += 1

        # сортировка словаря по значению
        sort_dict = sorted(dict_frequency.items(), key=operator.itemgetter(1), reverse=True)
        dict_frequency_df = pd.DataFrame(sort_dict, columns=['token', 'quantity'])

        dict_frequency_df['freq'] = dict_frequency_df['quantity'] / sum(dict_frequency_df['quantity'])
        dict_frequency_df.sort_values('freq', ascending=False)
        return dict_frequency_df


# класс для инициализации правил
# построение моделей для tf-idf, lsi, tf и параметров для этих моделей
# в этой версии добавляю возможность использовать последовательность файлов с синонимами
# kwargs["m"], kwargs["sinonims"], kwargs["ngrams"], kwargs["stopwords"], kwargs["dictionary"], kwargs["corpus"], kwargs["lingv_rules"]
# class RulesFill(TokensHandling, TextHandling):
class RulesFill(TextHandling):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.m = self.kwargs["m"]
        self.sinonims = self.kwargs["sinonims"]  # список синонимов и "канонических" форм
        self.ngrams = self.kwargs["ngrams"]  # список n-грамм
        self.stopwords = self.kwargs["stopwords"]  # стоп-слова
        self.dictionary = self.kwargs[
            "dictionary"]  # словарь токенизированный из коллекции (из соответствующего класса)
        self.corpus = self.kwargs["corpus"]  # корпус, если есть большой текстовый корпус
        self.logrules = self.logrules_fill()  # словарь лингвистических правил (тип правила и токены, которые к нему относятся) для каждого тега
        self.tfidf_parameters = self.tf_idf_data_fill()
        self.lsi_parameters = self.lsi_data_fill()
        self.tf_parameters = self.tf_data_fill()

    # функция оценивающая похожесть строк (возвращает оценку похожести)
    def strings_similarities(self, str1: str, str2: str):
        return difflib.SequenceMatcher(None, str1, str2).ratio()

    def logrules_fill(self):
        rules_df = self.kwargs["lingv_rules"]
        rules_df['lemm_words'] = rules_df['words'].apply(self.texts_handl)
        rules_df['working_text'] = rules_df['lemm_words'].apply(lambda tx: self.sinonims_handle(tx))
        rules_df['working_text'] = rules_df['working_text'].apply(
            lambda tx: self.patterns_change_split(self.ngrams, tx))
        rules_df['working_text'] = rules_df['working_text'].apply(lambda tx: self.stopwords_dell_split(tx))
        rules_dict = {}
        for tg in set(rules_df['tag']):
            rules_dict[tg] = list(
                zip(rules_df['rules'][rules_df['tag'] == tg], rules_df[rules_df['tag'] == tg]['working_text'],
                    rules_df[rules_df['tag'] == tg]['coeff']))
        return rules_dict

    # загрузим данные для модели tf-idf:
    # словарь и корпус для модели формируется на основании файла правил (из правил с признаком TF_IDF_SIM)
    def tf_idf_data_fill(self):
        # тексты для создания корпуса
        rules_corpus_texts = []
        for rule_num in self.logrules:
            for rule, txt, koeff in self.logrules[rule_num]:
                if rule == "TF_IDF_SIM":
                    rules_corpus_texts.append((rule_num, txt))
        try:
            self.dictionary.add_documents([tx.split() for rule_num, tx in rules_corpus_texts])
            # построим корпус, состоящий из векторов, соответствующих каждому правилу:
            rules_corpus = [(rule_num, self.dictionary.doc2bow(tx.split())) for rule_num, tx in rules_corpus_texts]
            rules, rulcorpus = zip(*rules_corpus)
            self.corpus = self.corpus + list(rulcorpus)
            tfidf_model = TfidfModel(self.corpus)
            return {"dictionary": self.dictionary, "model": tfidf_model, "rules_corpus": rules_corpus}
        except:
            return {"dictionary": None, "model": None, "rules_corpus": None}

    def lsi_data_fill(self, topics=500):
        rules_corpus_texts = []  # тексты для создания корпуса
        for rule_num in self.logrules:
            for rule, txt, koeff in self.logrules[rule_num]:
                if rule == "LSI_SIM":
                    rules_corpus_texts.append((rule_num, txt))
        try:
            self.dictionary.add_documents([tx.split() for rule_num, tx in rules_corpus_texts])
            # построим корпус, состоящий из векторов, соответствующих каждому правилу:
            rules_corpus = [(rule_num, self.dictionary.doc2bow(tx.split())) for rule_num, tx in rules_corpus_texts]
            rules, rulcorpus = zip(*rules_corpus)
            self.corpus = self.corpus + list(rulcorpus)
            lsi_model = LsiModel(self.corpus, id2word=self.dictionary, num_topics=topics)
            return {"dictionary": self.dictionary, "model": lsi_model, "rules_corpus": rules_corpus,
                    "num_topics": topics}
        except Exception:
            return {"dictionary": None, "model": None, "rules_corpus": None, "num_topics": None}

    # загрузим данные для модели tf:
    # словарь и корпус для модели формируется на основании файла правил (из правил с признаком TF_SIM)
    def tf_data_fill(self):
        # тексты для создания корпуса
        rules_corpus_texts = []
        for rule_num in self.logrules:
            for rule, txt, koeff in self.logrules[rule_num]:
                if rule == "TF_SIM":
                    rules_corpus_texts.append((rule_num, txt))
        try:
            self.dictionary.add_documents([tx.split() for rule_num, tx in rules_corpus_texts])
            # построим корпус, состоящий из векторов, соответствующих каждому правилу:
            rules_corpus = [(rule_num, self.dictionary.doc2bow(tx.split())) for rule_num, tx in rules_corpus_texts]
            rules, corpus = zip(*rules_corpus)
            return {"dictionary": self.dictionary, "rules_corpus": rules_corpus}
        except:
            return {"dictionary": None, "rules_corpus": None}


# класс для применения правил к заданному тексту
class TextsRulesApply(RulesFill):
    # m, sinonims, stopwords, ngrams, logrules, tf_parameters, tfidf_parameters, lsi_parameters, text: str)
    def __init__(self, text, **kwargs):
        self.kwargs = kwargs
        self.m = kwargs["m"]
        self.text = text
        self.sinonims = kwargs['sinonims']
        self.ngrams = kwargs["ngrams"]
        self.stopwords = kwargs["stopwords"]
        self.lemm_txt = self.text_prepare(
            self.text)  # список токенов, в которые превращается входящий (анализируемый) текст
        self.tfidf_indexes = self.tfidf_indexes_fill()
        self.tf_indexes = self.tf_indexes_fill()
        self.lsi_indexes = self.lsi_indexes_fill()
        self.tfidf_vector = self.tfidf_vector_fill()
        self.tf_vector = self.tf_vector_fill()
        self.lsi_vector = self.lsi_vector_fill()
        self.logrules_results = self.rules_results()

    # построение вектора tf-idf модели из полученного текста
    def tfidf_indexes_fill(self):
        try:
            dct = self.kwargs["tfidf_parameters"]["dictionary"]
            tfidf_model = self.kwargs["tfidf_parameters"]["model"]
            rules, corpus = zip(*self.kwargs["tfidf_parameters"]["rules_corpus"])
            txt_corp = dct.doc2bow(self.lemm_txt.split())
            txt_tf_idf_vect = tfidf_model[txt_corp]
            corpus_tf_idf_vects = [tfidf_model[x] for x in corpus]
            index = Similarity(None, corpus_tf_idf_vects, num_features=len(dct))
            rules_similarity = list(zip(rules, index[txt_tf_idf_vect]))
            return rules_similarity
        except:
            return None

    def tfidf_vector_fill(self):
        try:
            dct = self.kwargs["tfidf_parameters"]["dictionary"]
            tfidf_model = self.kwargs["tfidf_parameters"]["model"]
            rules, corpus = zip(*self.kwargs["tfidf_parameters"]["rules_corpus"])
            txt_corp = dct.doc2bow(self.lemm_txt.split())
            txt_tf_idf_vect = tfidf_model[txt_corp]
            return txt_tf_idf_vect
        except:
            return None

    # построение вектора lsi модели из полученного текста
    def lsi_indexes_fill(self):
        try:
            dct = self.kwargs["lsi_parameters"]["dictionary"]
            lsi_model = self.kwargs["lsi_parameters"]["model"]
            rules, corpus = zip(*self.kwargs["lsi_parameters"]["rules_corpus"])
            txt_corp = dct.doc2bow(self.lemm_txt.split())
            txt_vect = lsi_model[txt_corp]
            corpus_vects = [lsi_model[x] for x in corpus]
            index = Similarity(None, corpus_vects, num_features=self.kwargs["lsi_parameters"]["num_topics"])
            rules_similarity = list(zip(rules, index[txt_vect]))
            return rules_similarity
        except:
            return None

    def lsi_vector_fill(self):
        try:
            dct = self.kwargs["lsi_parameters"]["dictionary"]
            lsi_model = self.kwargs["lsi_parameters"]["model"]
            rules, corpus = zip(*self.kwargs["lsi_parameters"]["rules_corpus"])
            txt_corp = dct.doc2bow(self.lemm_txt.split())
            txt_vect = lsi_model[txt_corp]
            return txt_vect
        except:
            return None

    def tf_indexes_fill(self):
        try:
            dct = self.kwargs["tf_parameters"]["tf_dictionary"]
            rules, corpus = zip(*self.kwargs["tf_parameters"]["tf_rules_corpus"])
            txt_corp = dct.doc2bow(self.lemm_txt.split())
            index = Similarity(None, corpus, num_features=len(dct))
            rules_similarity = list(zip(rules, index[txt_corp]))
            return rules_similarity
        except:
            return None

    # построение вектора tf модели из полученного текста
    def tf_vector_fill(self):
        try:
            dct = self.kwargs["tf_parameters"]["tf_dictionary"]
            rules, corpus = zip(*self.kwargs["tf_parameters"]["tf_rules_corpus"])
            txt_corp = dct.doc2bow(self.lemm_txt.split())
            return txt_corp
        except:
            return None

    # функции, определяющие правила:
    def Include_AND(self, tokens_list, text_list):
        for token in tokens_list:
            if token not in text_list:
                return False
        return True

    def Include_OR(self, tokens_list, text_list):
        for token in tokens_list:
            if token in text_list:
                return True
        return False

    def Exclude_AND(self, tokens_list, text_list):
        for token in tokens_list:
            if token in text_list:
                return False
        return True

    def Exclude_OR(self, tokens_list, text_list):
        for token in tokens_list:
            if token not in text_list:
                return True
        return False

    # функция, анализирующая на вхождение в текст строки (последовательности токенов, а не токенов по-отдельности)
    def Include_STR(self, tokens_str, text_str):
        if tokens_str in text_str:
            return True
        else:
            return False

    def Exclude_STR(self, tokens_str, text_str):
        if tokens_str not in text_str:
            return True
        else:
            return False

    def Include_STR_P(self, tokens_list: list, txt_list: list, coeff):
        length = len(tokens_list)
        txts_split = [txt_list[i:i + length] for i in range(0, len(txt_list), 1) if
                      len(txt_list[i:i + length]) == length]
        for tx_l in txts_split:
            if self.strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff:  # self.sims_score:
                return True
        return False

    def Exclude_STR_P(self, tokens_list: list, txt_list: list, coeff):
        length = len(tokens_list)
        txts_split = [txt_list[i:i + length] for i in range(0, len(txt_list), 1) if
                      len(txt_list[i:i + length]) == length]
        for tx_l in txts_split:
            if self.strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff:  # self.sims_score:
                return False
        return True

    # функция, оценивающая близость на основании tf-idf модели
    def TF_IDF_SIM(self, rule_tag, coeff):
        for rl_num, tf_idf_coeff in self.tfidf_indexes:
            if rl_num == rule_tag and tf_idf_coeff >= coeff:
                return True
        return False

    def LSI_SIM(self, rule_tag, coeff):
        for rl_num, lsi_coeff in self.lsi_indexes:
            if rl_num == rule_tag and lsi_coeff >= coeff:
                return True
        return False

    # функция, оценивающая близость на основании tf модели
    def TF_SIM(self, rule_tag, coeff):
        for rl_num, tf_coeff in self.tf_indexes:
            if rl_num == rule_tag and tf_coeff >= coeff:
                return True
        return False

    # функция, анализирующая логико-лингвистические правила и возвращающая "ИСТИНА" или "ЛОЖЬ" по итогу (работает с фрагментом текста)
    def decision_rules(self, rule_tag, rule_tokens_coeff):
        decision = True
        for rl, tks, coeff in rule_tokens_coeff:
            if rl == 'Include_AND':
                quotient_decision = self.Include_AND(tks.split(), self.lemm_txt.split())
            elif rl == 'Include_OR':
                quotient_decision = self.Include_OR(tks.split(), self.lemm_txt.split())
            elif rl == 'Exclude_AND':
                quotient_decision = self.Exclude_AND(tks.split(), self.lemm_txt.split())
            elif rl == 'Exclude_OR':
                quotient_decision = self.Exclude_OR(tks.split(), self.lemm_txt.split())
            elif rl == 'Include_STR':
                quotient_decision = self.Include_STR(tks, self.lemm_txt)
            elif rl == 'Exclude_STR':
                quotient_decision = self.Exclude_STR(tks, self.lemm_txt)
            elif rl == 'Include_STR_P':
                quotient_decision = self.Include_STR_P(tks.split(), self.lemm_txt.split())
            elif rl == 'Exclude_STR_P':
                quotient_decision = self.Exclude_STR_P(tks.split(), self.lemm_txt.split())
            elif rl == 'TF_IDF_SIM':
                quotient_decision = self.TF_IDF_SIM(rule_tag, coeff)
            elif rl == 'TF_SIM':
                quotient_decision = self.TF_SIM(rule_tag, coeff)
            elif rl == 'LSI_SIM':
                quotient_decision = self.LSI_SIM(rule_tag, coeff)
            decision = decision and quotient_decision
        return decision

        # функция, возвращающая результат применения лингвистических правил:

    def rules_results(self):
        return [(tg, self.decision_rules(tg, self.kwargs["logrules"][tg])) for tg in self.kwargs["logrules"]]


if __name__ == "__main__":
    data_rout = r'./data'
    # data_rout = r'./data/3'
    stopwords_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'stopwords.csv'))
    lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'lingv_rules.csv'))
    sinonims_files = ['01_sinonims.csv', '02_sinonims.csv']

    sinonims_dfs = []
    for file_name in sinonims_files:
        sinonims_dfs.append(pd.read_csv(os.path.join(data_rout, 'bss_data', file_name)))

        # TokensHandling
    # переменные: kwargs["stopwords"], kwargs["sinonims"], kwargs["ngrams"]
    ngrams_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'ngrams.csv'))
    texts_collection_df = pd.read_csv(os.path.join(data_rout, 'bss_data', 'texts_collection.tsv'), sep='\t')

    kwargs = {"sinonims": sinonims_dfs, "stopwords": stopwords_df, "ngrams": ngrams_df}
    tkclass = TokensHandling(**kwargs)
    print(tkclass.sinonims)
    print(tkclass.stopwords)

    # TextHandling
    # проверка работы класса, лемматизирующего тексты с использованием имеющихся стоп-слов и синонимов (из класса TokensHandling)
    sinonims = tkclass.sinonims
    stopwords = tkclass.stopwords
    ngrams = tkclass.ngrams
    m = tkclass.m
    texts = ['кто должен должен сдает бухгалтерскую упрощенную отчетность с середины года физическое лицо',
             'которое просто хотя ндс сдавать бурый']

    kwargs = {"m": m, "sinonims": sinonims, "stopwords": stopwords, "ngrams": ngrams}
    txtclass = TextHandling(texts, **kwargs)
    print(txtclass.lemm_texts)

    # CollectionStatHandling
    # kwargs["texts_collection"] - список текстов
    texts_collection_df = pd.read_csv(os.path.join(data_rout, 'bss_data', "texts_collection.tsv"), sep='\t')
    texts_collection = list(texts_collection_df["texts"])
    kwargs = {"m": m, "sinonims": sinonims, "stopwords": stopwords, "ngrams": ngrams}
    clh = CollectionStatHandling(texts_collection, **kwargs)
    print(clh.bigrams_df)
    print(clh.gensim_corpus)
    print(clh.dict_frequency_df)
    print(len(clh.gensim_dictionary))

    # использование автоматичесикх биграмм
    bigr_df = clh.bigrams_df
    bigr_df = bigr_df[bigr_df["estimate"] > 2]

    # сформируем биграммы:
    ngrams = [" ".join([t1, t2]) for t1, t2 in zip(bigr_df['w1'], bigr_df['w2'])]
    tokens = list(bigr_df['bigrams'])
    ngrams_df = pd.DataFrame(list(zip(ngrams, tokens)), columns=['ngrams', 'token'])
    print(ngrams_df)

    # TextHandling с биграммами
    # проверка работы класса, лемматизирующего тексты с использованием имеющихся стоп-слов и синонимов (из класса TokensHandling)
    kwargs = {"sinonims": sinonims_dfs, "stopwords": stopwords_df, "ngrams": ngrams_df}
    tkclass = TokensHandling(**kwargs)

    # print(tkclass.sinonims)
    # print(tkclass.stopwords)
    sinonims = tkclass.sinonims
    stopwords = tkclass.stopwords
    ngrams = tkclass.ngrams
    print(ngrams)

    m = tkclass.m
    texts = [
        'кто должен должен сдает бухгалтерскую упрощенную отчетность с середины года физическое лицо счетов фактур',
        'которое просто хотя ндс сдавать бурый нарушение срока в таможенную службу']

    kwargs = {"m": m, "sinonims": sinonims, "stopwords": stopwords, "ngrams": ngrams}
    txtclass = TextHandling(texts, **kwargs)
    print(txtclass.lemm_texts)

    '''
    #RulesFill-1 - с корпусом и со словарем из большой коллекции
    data_rout = r'./data/1'
    lingv_rules = pd.read_csv(os.path.join(data_rout, "lingv_rules.csv"))
    #kwargs["m"], kwargs["sinonims"], kwargs["ngrams"], kwargs["stopwords"], kwargs["dictionary"], kwargs["corpus"], kwargs["lingv_rules"]
    kwargs = {"m" : m, "sinonims" : sinonims, "ngrams" : ngrams, "stopwords" : stopwords, "dictionary" : clh.gensim_dictionary, "corpus" : clh.gensim_corpus, "lingv_rules" : lingv_rules}
    rlh = RulesFill(**kwargs)
    print(rlh.logrules)

    #TextsRulesApply - 1
    text = 'кто должен Должен сдает бухгалтерскую Упрощенную отчетность с середины года Физическое лицо счет фактура'
    kwargs = {"m" : m, "logrules" : rlh.logrules, "sinonims" : sinonims, "ngrams" : ngrams, "stopwords" : stopwords, "dictionary" : clh.gensim_dictionary,
                 "corpus" : clh.gensim_corpus, "lingv_rules" : lingv_rules, "tfidf_parameters" : rlh.tfidf_parameters, "tf_parameters" : rlh.tf_parameters,
                  "lsi_parameters" : rlh.lsi_parameters}
    trapp = TextsRulesApply(text, **kwargs)
    print(trapp.logrules_results)

    #RulesFill-2 - без корпуса и словаря
    data_rout = r'./data/3'
    lingv_rules = pd.read_csv(os.path.join(data_rout, "lingv_rules.csv"))
    kwargs = {"m" : m, "sinonims" : sinonims, "ngrams" : ngrams, "stopwords" : stopwords, "dictionary" : None, "corpus" : None, "lingv_rules" : lingv_rules}
    rlh = RulesFill(**kwargs)
    print(rlh.logrules)
    
    #TextsRulesApply - 2 
    text = 'кто должен Должен сдает бухгалтерскую Упрощенную отчетность с середины года Физическое лицо'
    kwargs = {"m" : m, "logrules" : rlh.logrules, "sinonims" : sinonims, "ngrams" : ngrams, "stopwords" : stopwords, "dictionary" : clh.gensim_dictionary,
                 "corpus" : clh.gensim_corpus, "lingv_rules" : lingv_rules, "tfidf_parameters" : rlh.tfidf_parameters, "tf_parameters" : rlh.tf_parameters,
                  "lsi_parameters" : rlh.lsi_parameters}
    trapp = TextsRulesApply(text, **kwargs)
    print(trapp.logrules_results)
    '''
