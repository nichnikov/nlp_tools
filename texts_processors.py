import os, pickle, time, copy
from abc import ABC, abstractmethod
from utility import *

# абстрактный класс, определяющие методы для токенизаторов
class AbstractTokenizer(ABC):
    #@abstractmethod
    #def dict_tokenizer():
    #    pass

    @abstractmethod
    def texts_processing():
        pass

    @abstractmethod
    def model_tokenize():
        pass


# любой текст должен превращаться в набор токенов и все функции и классы работают с набором токенов
# после замены функции на класс, скорость обработки списка текстов уменьшилась с 82 сек. до 0.5 сек 
# за счет отказа от вызова Mystem для каждого текста
class TextsLematizer():
    def __init__(self):
        self.m = Mystem()

    # функция, проводящая предобработку текста 
    def text_hangling(self, text: str):
        try:
            txt = re.sub('[^a-zа-я\d]', ' ', text.lower())
            txt = re.sub('\s+', ' ', txt)
            # сюда можно будет вложить самую разную обработку, в том числе и вариационную
            return txt
        except:
            return ""

    # функция лемматизации одного текста
    def text_lemmatize(self, text: str):
        try:
            lemm_txt = self.m.lemmatize(text)
            lemm_txt = [w for w in lemm_txt if w not in [' ', '\n']]
            return lemm_txt
        except:
            return ['']

    # функция лемматизации списка текстов текста
    def texts_lemmatize(self, texts_list):
        return [self.text_lemmatize(self.text_hangling(tx)) for tx in texts_list]


# класс, в котором собраны функции обработки текстов (просто для удобства компановки)
# можно было бы обойтись определением этих функций (впрочем, теоретически сюда можно добавить что-то)
# например, какие-нибудь библиотеки nltk
class TextsProcessor(TextsLematizer):
    # токенезирует входящий список текстов
    # asc_dsc_tuples : [(),()...]
    def texts_asc_dsc_change(self, asc_dsc_tuples : [], splited_texts : [[]]):

        def patterns_change(asc_dsc_tuples, splited_text : []):
            for asc, dsc in [asc_dsc_tuples]:
                splited_text = replaceAscriptor(splited_text, asc, dsc)
            return splited_text

        return [patterns_change(asc_dsc_tuples, splited_text) for splited_text in splited_texts]

    # функция, применяющая лемматизатор к вложенным спискам:
    # dictionaries : [[]]
    def dictionaries_tokenizer(self, dictionaries, toknize = True):
        # для случая, когда словари нужно лемматизировать:
        if toknize == True:
            dictionaries_lemm = []
            for dictionary in dictionaries:
                temp_dict_lemm = []                
                # лемматизация синонимов ([([], []), ([], []), ...]):
                if dictionary != [] and isinstance(dictionary[0], tuple):
                    words, sinonims = zip(*dictionary)  
                    temp_dict_lemm.append(list(zip(self.texts_lemmatize(words), self.texts_lemmatize(sinonims))))
                elif isinstance(dictionary, list) and dictionary != []:
                    temp_dict_lemm.append(self.texts_lemmatize(dictionary))
                elif isinstance(dictionary, list) and dictionary == []:
                    temp_dict_lemm.append(dictionary)
                #dictionaries_lemm.append(temp_dict_lemm)
                dictionaries_lemm = dictionaries_lemm + temp_dict_lemm
        # для случая, когда словари лемматизировать не нужно (но нужно привести к виду [[token1, token2, ...]] или [([asc: tk1, tk2, ...], [des: tk1, tk2, ...]), ...):
        else:
            dictionaries_lemm = []
            for dictionary in dictionaries:
                temp_dict_lemm = []
                if dictionary != [] and isinstance(dictionary[0], tuple):
                    words, sinonims = zip(*dictionary)
                    temp_dict_lemm.append(list(zip([w.split() for w in words], [s.split() for s in sinonims])))
                elif isinstance(dictionary, list) and dictionary != []:
                    temp_dict_lemm.append([w.split() for w in dictionary])
                elif isinstance(dictionary, list) and dictionary == []:
                    temp_dict_lemm.append(dictionary)
                #dictionaries_lemm.append(temp_dict_lemm)
                dictionaries_lemm = dictionaries_lemm + temp_dict_lemm

        return dictionaries_lemm

    def txts_lemmatize(self, incoming_texts):
        output_texts = []
        for incoming_text in incoming_texts:
            lemm_tx = self.texts_lemmatize([incoming_text])
        return lemm_tx

    def texts_asc_dsc_ch(self, lemm_tx, dictionaries_lemm):
        # замена синонимов в тексте:
        for asc_dsc_tuple_list in dictionaries_lemm:
            lemm_tx = self.texts_asc_dsc_change(asc_dsc_tuple_list, lemm_tx)        
        return lemm_tx

    def texts_stowords_dell(self, lemm_tx, stopwords_list):
        # удаление стоп-слов:
        if stopwords_list != []:
            lemm_tx = [[w for w in tx if [w] not in stopwords_list] for tx in lemm_tx]
        return lemm_tx

    def texts_workwords_apply(self, lemm_tx, stopwords_list):
        # оставление только значимых слов:
        if stopwords_list != []:
            lemm_tx = [[w for w in tx if [w] in workwords_list] for tx in lemm_tx]
        return lemm_tx

# простой токенизатор (лемматизирует словари и применяет лемматизацию и словари к входящим текстам)
class SimpleTokenizer(AbstractTokenizer, TextsProcessor):    
    def __init__(self, loader_obj):
        self.m = Mystem() # по непонятной причине функция из другого класса, в котором m определена, не видит в этом классе Майстем (раньше уже была такая проблема)
        self.dict_types = [("sinonims", self.texts_asc_dsc_ch), ("ngrams", self.texts_asc_dsc_ch), 
                ("stopwords", self.texts_stowords_dell), ("workwords", self.texts_workwords_apply)] #список обязательных ключей к словарям, от них зависит логика применения словарей
        self.model = loader_obj # передается не "сырая модель", а объект класса Лоадер
        #assert "SimpleTokenizer" in self.model.tokenizer_type, ("тип токенизатора, определенное в классе SimpleTokenizer не соответствует типу токенизатора в модели") # проверка, что в модели прописан именно этот токенайзер:
        self.dictionaries_lemm = self.dict_tokenizer()

    # токенизация словарей (лемматизация)
    def dict_tokenizer(self):
        lemm_dicts = []
        model_dicts_list = self.model.dictionaries
        for dict_ in model_dicts_list:
            tokenize = dict_["tokenize"]
            temp_dict = {}
            for dict_name in dict_:
                if dict_name != "tokenize":
                #assert dict_name in self.dict_types
                    temp_dict[dict_name] = self.dictionaries_tokenizer(dict_[dict_name], tokenize)
            lemm_dicts.append(temp_dict)
        return lemm_dicts

    def texts_processing(self, incoming_texts):

        output_texts = []
        for incoming_text in incoming_texts:
            lemm_tx = self.texts_lemmatize([incoming_text])                        
            
            # применение лингвистики:
            for dict_ in self.dictionaries_lemm:
                for dict_name in dict_:
                    for func_name, func in self.dict_types:
                        if dict_name == func_name:
                            for obj in dict_[dict_name]:
                                lemm_tx = func(lemm_tx, obj)
            
            output_texts = output_texts + lemm_tx        

        return output_texts
    
    # функция, возвращающая токенизированнную модель 
    def model_tokenize(self):
        tkn_model = copy.copy(self.model)
        # токенизация словарей
        tkn_model.dictionaries = self.dictionaries_lemm
        # токенизация эталонов
        tkn_apl_field = {}
        for name in self.model.application_field:
            if name == "texts":
                tkn_apl_field[name] = self.texts_processing(self.model.application_field[name])
            else:
                tkn_apl_field[name] = self.model.application_field[name]
        tkn_model.application_field = tkn_apl_field
        return tkn_model

class Doc2VecTokenizer(AbstractTokenizer):
    def __init__(self, loader_obj):
        # получим все тексты с первоночальной токенизацией (лемматизация, словари и т. п.):
        #assert "Doc2VecTokenizer" in loader_obj.tokenizer_type, ("тип токенизатора, определенное в классе SimpleTokenizer не соответствует типу токенизатора в модели") # проверка, что в модели прописан именно этот токенайзер:
        self.simple_tokenizer = SimpleTokenizer(loader_obj)
        self.simple_tokenize_model = self.simple_tokenizer.model_tokenize()
        
    def texts_processing(self, texts):
        initial_tokenize_texts = self.simple_tokenizer.texts_processing(texts)
        return [self.simple_tokenize_model.texts_algorithms["d2v_model"].infer_vector(tk_tx) for tk_tx in initial_tokenize_texts]

    def model_tokenize(self):
        tkn_model = copy.copy(self.simple_tokenize_model)
        # токенизация словарей
        None
        # токенизация эталонов
        tkn_apl_field = {}
        for name in tkn_model.application_field:
            if name == "texts":
                tkn_apl_field[name] = [self.simple_tokenize_model.texts_algorithms["d2v_model"].infer_vector(tk_tx) for tk_tx in tkn_model.application_field["texts"]]
            else:
                tkn_apl_field[name] = self.simple_tokenize_model.application_field[name]
        tkn_model.application_field = tkn_apl_field
        return tkn_model

# с обнавлением словарей и без
class LsiTokenizer(AbstractTokenizer):
    def __init__(self, loader_obj):
        # получим все тексты с первоночальной токенизацией (лемматизация, словари и т. п.):
        self.model = loader_obj
        self.simple_tokenizer = SimpleTokenizer(self.model)
        self.simple_tokenize_model = self.simple_tokenizer.model_tokenize()

    # построение LSI вектора из произвольного лемматизированного и токенизированного текста
    def lsi_text2vector(self, texts : [[]]):
        lsi_vectors = []
        for text in texts:
            txt_corp = self.model.texts_algorithms["dictionary"].doc2bow(text)
            txt_vect = self.model.texts_algorithms["model"][txt_corp]
            lsi_vectors.append(txt_vect)
        return lsi_vectors
        
    def texts_processing(self, texts):
        initial_tokenize_texts = self.simple_tokenizer.texts_processing(texts)
        return self.lsi_text2vector(initial_tokenize_texts)

    def model_tokenize(self):
        tkn_model = copy.copy(self.simple_tokenize_model)
        # токенизация словарей
        None
        # токенизация эталонов
        tkn_apl_field = {}
        for name in tkn_model.application_field:
            if name == "texts":
                tkn_apl_field[name] = self.lsi_text2vector(tkn_model.application_field["texts"])
            else:
                tkn_apl_field[name] = self.simple_tokenize_model.application_field[name]
        tkn_model.application_field = tkn_apl_field
        return tkn_model

# класс, получающий на вход модель и умеющий ее токенизировать и токенизировать тексты в соответствие с моделью
# данный класс "знает", какой токенизатор какой модели соответствует и умеет применять нужный
# пользователи, которым нужна токенизация (токенизированные тексты и модели) имеют дело с этим классом
class TokenizerApply(AbstractTokenizer):
    def __init__(self, loader_obj):
        self.tknz_types = [("SimpleTokenizer", SimpleTokenizer), ("Doc2VecTokenizer", Doc2VecTokenizer), 
                    ("LsiTokenizer", LsiTokenizer)]        
        self.model = loader_obj    
        # проверка соответствия имени 
        #assert self.model.tokenizer_type in [x[0] for x in self.tknz_types], ("имя tokenizer_type в классе Loader не соответствует именам self.tknz_types")

        for tk_type, TknzClass in self.tknz_types:
            if tk_type == self.model.tokenizer_type:
                self.tnzr = TknzClass(self.model)

    def model_tokenize(self):
        return self.tnzr.model_tokenize()

    def texts_processing(self, incoming_text):
        return self.tnzr.texts_processing(incoming_text)
                

if __name__ == "__main__":
    data_rout = r'./data'
    models_rout = r'./models'
    
    """
    with open(os.path.join(models_rout, "fast_answrs", "include_and_model.pickle"), "br") as f:
        model = pickle.load(f)    
    smpltk = SimpleTokenizer(Loader(model)) 
    """

    txts = ["упрощенная бухгалтерская отчетность кто сдает Фи ТАм котОРый али бы", "кто должен сдавать аудиторское заключение", "кто должен подписывать справки", "парит летит воздушный судно"]

    with open(os.path.join(models_rout, "fast_answrs", "lsi_model.pickle"), "br") as f:
        model = pickle.load(f)

    '''
    lsi_tkz = LsiTokenizer(Loader(model))
    t1 = time.time()
    tk_m = lsi_tkz.model_tokenize()
    print(time.time() - t1)
    
    tk_txt = lsi_tkz.texts_processing(txts)
    print(tk_txt)
    print(len(tk_txt))
    '''

    tk_appl = TokenizerApply(Loader(model))
    print(tk_appl.texts_processing(txts))