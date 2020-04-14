# здесь будут функции, которые нужны для других объектов
import os, re, pickle, difflib, keras
from pymystem3 import Mystem
from abc import ABC, abstractmethod
from keras import backend as K

# определяет тип модели для сервиса:
class AbstractLoader(ABC):
    def __init__(self):
        self.model_type = None
        self.classificator_algorithms = None
        self.texts_algorithms = None
        self.dictionaries = None
        self.tokenizer_type = None
        self.application_field = None

# данный класс (или функция) должен знать все форматы моделей и уметь их загружать
class Loader(AbstractLoader):
    def __init__(self, incoming_model):        
        self.in_model = self.model_load(incoming_model)
        
        self.model_type = self.in_model["model_type"] # возвращает ключ модели (имя модели) по ключу остальные объекты "понимают" что за функции им нужно запускать                
        
        self.classificator_algorithms = self.in_model["classificator_algorithms"] # возвращает модели для правил
        
        self.texts_algorithms = self.in_model["texts_algorithms"] # возвращает модели для обработки текстов (например, Word2Vec - модели векторизации)
        
        self.dictionaries = self.in_model["lingvo"] # возвращает словари для обработки текста
        
        self.tokenizer_type = self.in_model["tokenizer"] # возвращает тип токенезации входящего текста

        self.application_field = self.in_model["etalons"]

    
    # загрузка модели с проверкой имен:
    def model_load(self, incoming_model):        
        names = [nm for nm in incoming_model]
        for nm in names:
            assert nm in ["model_name", "model_type", 
                "etalons", "texts", "coeff", "tags", "lingvo", 
                "classificator_algorithms", "texts_algorithms", "tokenizer"],  ("наименования входящей модели не соответствуют ожиданию класса Loader")

        assert incoming_model["model_type"] in ["siamese_lstm_d2v", "simple_rules", "lsi", "intersec_share"], ("тип модели не соответствует ожиданию класса Loader")    

        etalons_dict_names = [nm for nm in incoming_model["etalons"]]
        # возвращает эталоны, к которым применяется правило (проверяет их на соответствие соглашению)
        for nm in etalons_dict_names:
            assert nm in ["rules", "texts", "coeff", "tags"], ("имена словаря etalons не соответствуют ожиданиям класса Loader")

        return incoming_model

""" Утилитарные функции над массивами """
""" Оставляет только базовые элементы сложной (иерархической) структуры """
def flatten_all(iterable):
    for elem in iterable:
        if not isinstance(elem, list):
            yield elem
        else:
            for x in flatten_all(elem):
                yield x

def flatten_all_tuple(iterable):
    for elem in iterable:
        if not isinstance(elem, tuple):
            yield elem
        else:
            for x in flatten_all_tuple(elem):
                yield x


"""Нарезает массив окном размера len c шагом stride"""
def sliceArray(src:[], length:int=1, stride:int=1):
    return [src[i:i+length] for i in range(0, len(src), stride) if len(src[i:i+length]) == length]

'''Нарезает массив окном размера len с шагом в размер окна'''
def splitArray(src:[], length:int):
    return sliceArray(src, length, length)

"""Преобразует массив токенов в мешок (каждый токен представлен кортежем -- (токен, сколько раз встречается в массиве))"""
def arr2bag(src:[]):
    return [(x, src.count(x)) for x in set(src)]

"""Возвращает массив токенов src за исключением rem"""
def removeTokens(src:[], rem:[]):    
    return [t for t in src if t not in rem]

"""Заменяет в массиве src множество токенов аскриптора asc дексрипторами токена (синонимия)"""
def replaceAscriptor(src:[], asc:[], desc:[]):
    src_repl = []
    length = len(asc)
    src_ = [src[i:i+length] for i in range(0, len(src), 1)]
    i = 0
    while i < len(src_):
        if src_[i] == asc:
            src_repl = src_repl + desc
            i+=length
        else:
            src_repl.append(src_[i][0])
            i+=1
    return src_repl

""" Прочие функции """
""" функция оценивающая похожесть строк (возвращает оценку похожести) """
def strings_similarities(str1 : str, str2 : str):
    return difflib.SequenceMatcher(None, str1, str2).ratio()

# сравнение двух списков (возвращает токены, принадлежащие обоим спискам)
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

#df["text"]
# лемматизация текстов датафрейма
# columns_for_change_names - список столбцов (имен столбцов) датафрейма, которые должны быть обработаны
# changed_columns_names - список столбцов (имен столбцов) датафрейма после обработки
def df_lemmataze(df, columns_for_change_names = ["text"], changed_columns_names = ["changed_text"]):
    assert(len(columns_for_change_names) == len(changed_columns_names))
    for column_for_ch, ch_column in zip(columns_for_change_names, changed_columns_names):
        df[ch_column] = df[column_for_ch].apply(lambda tx: texts_lemmatize([tx]))
    return df

# замена словарей текстов датафрейма
# columns_for_change_names : [[]]
#asc_dsc_tuples_list : []
def df_processing(df, asc_dsc_tuples_list, columns_for_change_names = ["text"], changed_columns_names = ["changed_text"]):
    assert(len(columns_for_change_names) == len(changed_columns_names))
    for column_for_ch, ch_column in zip(columns_for_change_names, changed_columns_names):
        df[ch_column] = df[column_for_ch]
        for asc_dsc_tuple in asc_dsc_tuples_list:
            df[ch_column] = df[ch_column].apply(lambda tx: texts_asc_dsc_change(asc_dsc_tuple, tx))
    return df

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

keras.losses.contrastive_loss = contrastive_loss

if __name__ == "__main__":
    data_rout = r'./data'
    models_rout = r'./models'

    with open(os.path.join(models_rout, "fast_answrs","include_and_model.pickle"), "br") as f:
        model = pickle.load(f)
    
    ld = Loader(model)

    print(ld.application_field)

    ab = AbstractLoader()
    print(ab)
    print(ab.application_field)
    ab.application_field = "я помню чудное..."
    print(ab.application_field)

   
    
