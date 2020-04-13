'''
идея в том, чтобы создать класс, объединяющий в себе методы обработки текста (замену синонимов, замену n-грамм) 
и данные (массив синонимов и n-грамм, набор правил) для такой обработки
Пока не ясно, что должно являться экземпляром класса и на каком 
'''
#%%
import os, re, pickle, difflib, time
from pymystem3 import Mystem
import pandas as pd
import numpy as np
from gensim import similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import Similarity

""" Утилитарные функции над массивами """
"""Нарезает массив окном размера len c шагом stride"""
def flatten_all(iterable):
    for elem in iterable:
        if not isinstance(elem, list):
            yield elem
        else:
            for x in flatten_all(elem):
                yield x


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

#класс для инициализации правил
#в этой версии добавляю возможность использовать последовательность файлов с синонимами
class RulesFill():
    def __init__ (self, data_rout, sinonims_files):        
        self.m = Mystem()
        self.data_rout = data_rout #путь к данным (к файлам csv с синонимами, правилами, словарем и n-граммами)
        self.sinonims_files = sinonims_files #список файлов, содержащих синонимы в той последовательности, в которой их нужно применять
        self.sinonims = self.sinonims_fill() #список синонимов и "канонических" форм
        self.ngrams = self.ngrams_fill() #список n-грамм 
        self.stopwords = self.stopwords_fill()
        self.logrules = self.logrules_fill() #словарь лингвистических правил (тип правила и токены, которые к нему относятся) для каждого тега
        self.tfidf_parameters = self.tf_idf_data_fill()
        self.tf_parameters = self.tf_data_fill()
    
    #функция оценивающая похожесть строк (возвращает оценку похожести)
    def strings_similarities(self, str1 : str, str2 : str):
        return difflib.SequenceMatcher(None, str1, str2).ratio()
    
    #функция, проводящая предобработку текста (включая лемматизацию)    
    def texts_handl(self, txt: str):
        lemm_txt = ' '.join(self.m.lemmatize(txt.lower()))
        lemm_txt = re.sub('[^a-zа-я\d]', ' ', lemm_txt)
        lemm_txt = re.sub('\s+', ' ', lemm_txt)
        return lemm_txt   
    
    #надо поменять на работу с токенами (отказаться от регулярных выражений):
    def patterns_change(self, sinonims_initial_forms_tuples, text_for_change : str):
        for sinonim, initial_form in sinonims_initial_forms_tuples:
            text_for_change = re.sub(sinonim, ' '+initial_form+' ', text_for_change)
        return text_for_change

    #надо поменять на работу с токенами (отказаться от регулярных выражений):
    def patterns_change_split(self, asc_dsc_tuples, text_for_change : str):
        for asc, dsc in asc_dsc_tuples:
            text_for_change = ' '.join(replaceAscriptor(text_for_change.split(), asc.split(), dsc.split()))
        return text_for_change

    def stopwords_dell(self, stop_words, text_for_change : str):
        for stopword in stop_words:
            text_for_change = re.sub('\b'+stopword+'\b', ' ', text_for_change)
        return text_for_change

    #замена стоп-слов без регулярных выражений, удаление токенов
    def stopwords_dell_split(self, stop_words, text_for_change : str):
        for stopword in stop_words:
            text_for_change = ' '.join([x for x in flatten_all(replaceAscriptor(text_for_change.split(), stopword.split(), ['']))])            
        return text_for_change

    def stopwords_fill(self):
        stopwords_df = pd.read_csv(os.path.join(self.data_rout, 'stopwords.csv'))
        stopwords_df['lemm_words'] = stopwords_df['words'].apply(self.texts_handl)
        stopwords_df['working_text']  = stopwords_df['lemm_words'].apply(lambda tx: self.sinonims_handle(self.sinonims, tx))
        stopwords_df['working_text'] = stopwords_df['working_text'].apply(lambda tx: self.patterns_change_split(self.ngrams, tx))
        return list(stopwords_df['working_text'])
    
    def sinonims_fill(self):
        sinonims_list = []
        for file_name in self.sinonims_files:
            sinonims_df = pd.read_csv(os.path.join(self.data_rout, file_name))           
            sinonims_df['lemm_words'] = sinonims_df['words'].apply(self.texts_handl)
            sinonims_df['lemm_init_forms'] = sinonims_df['initial_forms'].apply(self.texts_handl)
            sinonims_list.append(list(zip(sinonims_df['lemm_words'], sinonims_df['lemm_init_forms'])))
        return sinonims_list

    def ngrams_fill(self):
        ngrams_df = pd.read_csv(os.path.join(self.data_rout, 'ngrams.csv'))
        ngrams_df['lemm_ngrams'] = ngrams_df['ngrams'].apply(self.texts_handl)
        return list(zip(ngrams_df['lemm_ngrams'],  ngrams_df['token']))
    
    #функция, замещающуя синонимы в тексте:
    def sinonims_handle(self, sinonims, text):
        for sin_tuples in sinonims:
            text = self.patterns_change_split(sin_tuples, text)
        return text

    def logrules_fill(self):
        rules_df = pd.read_csv(os.path.join(self.data_rout, 'lingv_rules.csv'))
        rules_df['lemm_words'] = rules_df['words'].apply(self.texts_handl)
        rules_df['working_text']  = rules_df['lemm_words'].apply(lambda tx: self.sinonims_handle(self.sinonims, tx))
        rules_df['working_text'] = rules_df['working_text'].apply(lambda tx: self.patterns_change_split(self.ngrams, tx))
        rules_df['working_text'] = rules_df['working_text'].apply(lambda tx: self.stopwords_dell_split(self.stopwords, tx))
        rules_dict = {}
        for tg in set(rules_df['tag']):
            rules_dict[tg] = list(zip(rules_df['rules'][rules_df['tag']==tg], rules_df[rules_df['tag']==tg]['working_text'], 
            rules_df[rules_df['tag']==tg]['coeff']))            
        return rules_dict

    #загрузим данные для модели tf-idf:
    #словарь и корпус для модели формируется на основании файла правил (из правил с признаком TF_IDF_SIM)
    def tf_idf_data_fill(self):
        rules_corpus_texts = [] #тексты для создания корпуса
        for rule_num in self.logrules:
            for rule, txt, koeff in self.logrules[rule_num]:
                if rule == "TF_IDF_SIM":
                    rules_corpus_texts.append((rule_num, txt))
        try:
            dct = Dictionary([tx.split() for rule_num, tx in rules_corpus_texts])
            #построим корпус, состоящий из векторов, соответствующих каждому правилу:
            rules_corpus = [(rule_num ,dct.doc2bow(tx.split())) for rule_num, tx in rules_corpus_texts]
            rules, corpus =  zip(*rules_corpus)
            tfidf_model = TfidfModel(corpus)
            return {"tf_idf_dictionary" : dct, "tfidf_model" : tfidf_model, "tf_idf_rules_corpus" : rules_corpus}
        except:            
            return {"tf_idf_dictionary" : None, "tfidf_model" : None, "tf_idf_rules_corpus" : None}

    #загрузим данные для модели tf:
    #словарь и корпус для модели формируется на основании файла правил (из правил с признаком TF_SIM)
    def tf_data_fill(self):
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


#класс для применения правил к заданному тексту
class TextsAnalysis(RulesFill):
    def __init__ (self, m, sinonims, stopwords, ngrams, logrules, tf_parameters, tfidf_parameters, text: str):
        self.m = m
        self.sinonims = sinonims
        self.stopwords = stopwords 
        self.ngrams = ngrams
        self.logrules = logrules 
        self.tfidf_parameters = tfidf_parameters
        self.tf_parameters = tf_parameters
        self.text = text
        self.lemm_txt = self.text_prepare() #список в которые превращается входящий (анализируемый) текст    
        self.tf_idf_vector = self.tfidf_text2vector()
        self.tf_vector = self.tf_text2vector()    
        self.logrules_results = self.rules_results()
       
    
    #построение вектора tf-idf модели из полученного текста
    def tfidf_text2vector(self):
        try:
            dct = self.tfidf_parameters["tf_idf_dictionary"]
            tfidf_model = self.tfidf_parameters["tfidf_model"]
            rules, corpus =  zip(*self.tfidf_parameters["tf_idf_rules_corpus"])
            txt_corp = dct.doc2bow(self.lemm_txt.split())
            txt_tf_idf_vect = tfidf_model[txt_corp]
            corpus_tf_idf_vects = [tfidf_model[x] for x in corpus]
            index = Similarity(None, corpus_tf_idf_vects, num_features=len(dct)) 
            rules_similarity = list(zip(rules, index[txt_tf_idf_vect]))
            return rules_similarity
        except:
            return None

    #построение вектора tf модели из полученного текста
    def tf_text2vector(self):
        try:
            dct = self.tf_parameters["tf_dictionary"]
            rules, corpus =  zip(*self.tf_parameters["tf_rules_corpus"])
            txt_corp = dct.doc2bow(self.lemm_txt.split())
            index = Similarity(None, corpus, num_features=len(dct)) 
            rules_similarity = list(zip(rules, index[txt_corp]))
            return rules_similarity
        except:
            return None

    #функция, которая возвращает текст, лемматизированный с замененными синонимами и n-граммами, приведенный к токенам из словаря:
    def text_prepare(self):
        #(1)лемматизация:
        lemm_tx = self.texts_handl(self.text)
        #(2)замена синонимов и n-грамм
        lemm_tx = self.sinonims_handle(self.sinonims, lemm_tx)
        lemm_tx = self.patterns_change_split(self.ngrams, lemm_tx) #замена на сплит
        lemm_tx = self.stopwords_dell_split(self.stopwords, lemm_tx)
        #вернем лемматизированный текст с замененными синонимами и n-граммами
        return lemm_tx
    
    #функции, определяющие правила:
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

    #функция, анализирующая на вхождение в текст строки (последовательности токенов, а не токенов по-отдельности)
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

    def Include_STR_P(self, tokens_list : list, txt_list : list, coeff):
        length = len(tokens_list)
        txts_split = [txt_list[i:i+length] for i in range(0, len(txt_list), 1) if len(txt_list[i:i+length]) == length]
        for tx_l in txts_split:
            if self.strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff: #self.sims_score:
                return True
        return False

    def Exclude_STR_P(self, tokens_list : list, txt_list : list, coeff):
        length = len(tokens_list)
        txts_split = [txt_list[i:i+length] for i in range(0, len(txt_list), 1) if len(txt_list[i:i+length]) == length]
        for tx_l in txts_split:
            if self.strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff: #self.sims_score:
                return False
        return True

    #функция, оценивающая близость на основании tf-idf модели
    def TF_IDF_SIM(self, rule_tag, coeff):
        for rl_num, tf_idf_coeff in self.tf_idf_vector:
            if rl_num == rule_tag and tf_idf_coeff >= coeff:
                return True
        return False

    #функция, оценивающая близость на основании tf модели
    def TF_SIM(self, rule_tag, coeff):
        for rl_num, tf_coeff in self.tf_vector:
            if rl_num == rule_tag and tf_coeff >= coeff:
                return True
        return False


    #функция, анализирующая логико-лингвистические правила и возвращающая "ИСТИНА" или "ЛОЖЬ" по итогу (работает с фрагментом текста)
    def decision_rules(self, rule_tag, rule_tokens_coeff):
        decision = True
        for rl, tks, coeff in rule_tokens_coeff:
            if rl ==  'Include_AND':
                quotient_decision = self.Include_AND(tks.split(), self.lemm_txt.split())
            elif rl ==  'Include_OR':
                quotient_decision = self.Include_OR(tks.split(), self.lemm_txt.split())
            elif rl ==  'Exclude_AND':
                quotient_decision = self.Exclude_AND(tks.split(), self.lemm_txt.split())
            elif rl ==  'Exclude_OR':
                quotient_decision = self.Exclude_OR(tks.split(), self.lemm_txt.split())
            elif rl ==  'Include_STR':
                quotient_decision = self.Include_STR(tks, self.lemm_txt)
            elif rl ==  'Exclude_STR':
                quotient_decision = self.Exclude_STR(tks, self.lemm_txt)
            elif rl ==  'Include_STR_P':
                quotient_decision = self.Include_STR_P(tks.split(), self.lemm_txt.split())
            elif rl ==  'Exclude_STR_P':
                quotient_decision = self.Exclude_STR_P(tks.split(), self.lemm_txt.split())
            elif rl ==  'TF_IDF_SIM':
                quotient_decision = self.TF_IDF_SIM(rule_tag, coeff)            
            elif rl ==  'TF_SIM':
                quotient_decision = self.TF_SIM(rule_tag, coeff)            
            decision = decision and quotient_decision
        return decision   
    
    #функция, возвращающая результат применения лингвистических правил:
    def rules_results(self):
        return [(tg, self.decision_rules(tg, self.logrules[tg])) for tg in self.logrules]

if __name__ == "__main__":    
    data_rout = r'./data/3'    
    rules_tool = RulesFill(data_rout, ['01_sinonims.csv', '02_sinonims.csv'])
    
    #print(rules_tool.sinonims)
    print(rules_tool.logrules)
    
    #сохранение правил после применения синонимов и н-грамм
    logrules = rules_tool.logrules 
    logrules_df = pd.DataFrame({})
    for lr in logrules:
        temp_df = pd.concat([pd.DataFrame([lr]*len(logrules[lr])), pd.DataFrame(logrules[lr])], axis = 1)
        logrules_df = pd.concat([logrules_df, temp_df])
    logrules_df.to_csv('logrules_after_handling.csv', index = False)

    
    m = rules_tool.m
    sinonims = rules_tool.sinonims
    stopwords = rules_tool.stopwords
    logrules = rules_tool.logrules
    tfidf_parameters = rules_tool.tfidf_parameters
    tf_parameters = rules_tool.tf_parameters
    ngrams = rules_tool.ngrams
    #txt = "кто сдает бухгалтерскую упрощенную отчетность"
    txt = 'желтый веселый крокодил'
    t0 = time.time()
    anals_tx = TextsAnalysis(m, sinonims, stopwords, ngrams, logrules, tf_parameters, tfidf_parameters, txt)
    t1 = time.time()
    #print(anals_tx.logrules_results)
    #print(anals_tx.sinonims)
    #print(anals_tx.lemm_txt)
    print(anals_tx.logrules_results)
    print(t1 - t0)
    
# %%
