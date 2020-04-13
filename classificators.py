# здесь будут объекты для создания правил
# https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
# https://kobkrit.com/tensor-something-is-not-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1

import os, pickle, time, sys
import numpy as np
from abc import ABC, abstractmethod
from utility import *
from texts_processors import *
import sys
import tensorflow as tf
from keras.optimizers import Adam

# объект SimpleTokenizer загружает в себя параметры, соответствующие модели и в дальнейшем в рамках этой модели
# в соответствие с загруженными параметрами происходит токенизация любых текстов
# преимущество объектного подхода перед функцией - объект создается один раз под модель (словари загружаются и обрабатываются один раз)
# затем многократно используются (данные и методы лемматизации заключены в объект)
# в случае использования функций, пришлось бы создавать отдельные переменные для хранения загруженных параметров


class AbstractRules(ABC):
    def __init__():
        # перменная, описывающая, какие модели входят в класс
        self.model_types = []

    @abstractmethod
    def rules_apply(self, text : []):
        pass

class SimpleRules(AbstractRules):
    def __init__(self, loader_obj):
        self.model_types = [("simple_rules", None)]
        self.functions_dict = {"include_and" : self.include_and, "include_or" : self.include_or,
        "exclude_and" : self.exclude_and, "exclude_or" : self.exclude_or,
        "include_str" : self.include_str, "include_str_p" : self.include_str_p,
        "exclude_str_p" : self.exclude_str_p, "intersec_share" : self.intersec_share}
        self.model = loader_obj
        self.tknzr = TokenizerApply(self.model)
        self.tknz_model = self.tknzr.model_tokenize()

    def rules_apply(self, texts):
        decisions = []
        # применим правило к токенизированным текстам:
        for num, tknz_tx in enumerate(self.tknzr.txts_processing(texts)):
            decisions_temp = []
            unique_tags = list(set(self.tknz_model.application_field["tags"]))
            for tg in unique_tags:
                decision = True
                for rule, tknz_etalon, tag, coeff in zip(self.tknz_model.application_field["rules"], self.tknz_model.application_field["texts"],
                    self.tknz_model.application_field["tags"], self.tknz_model.application_field["coeff"]):
                    if tag == tg:
                        decision = decision and self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)
                decisions_temp.append((tg, decision))
            decisions.append((num, decisions_temp))
        return decisions

    def include_and(self, tokens_list, text_list, coeff = 0.0):
        for token in tokens_list:
            if token not in text_list:
                return False
        return True

    def include_or(self, tokens_list, text_list, coeff = 0.0):
        for token in tokens_list:
            if token in text_list:
                return True
        return False

    def exclude_and(self, tokens_list, text_list, coeff = 0.0):
        for token in tokens_list:
            if token in text_list:
                return False
        return True

    def exclude_or(self, tokens_list, text_list, coeff = 0.0):
        for token in tokens_list:
            if token not in text_list:
                return True
        return False

    def intersec_share(self, tokens_list, text_list, intersec_coeff = 0.7): 
        intersec_tks = intersection(tokens_list, text_list)
        if len(intersec_tks)/len(tokens_list) > intersec_coeff:
            return True
        else:
            return False

    #функция, анализирующая на вхождение в текст строки (последовательности токенов, а не токенов по-отдельности)
    def include_str(self, tokens_str, text_str, coeff = 0.0):
        if tokens_str in text_str:
            return True
        else:
            return False

    def exclude_str(self, tokens_str, text_str, coeff = 0.0):
        if tokens_str not in text_str:
            return True
        else:
            return False

    def include_str_p(self, tokens_list : list, txt_list : list, coeff):
        length = len(tokens_list)
        txts_split = [txt_list[i:i+length] for i in range(0, len(txt_list), 1) if len(txt_list[i:i+length]) == length]
        for tx_l in txts_split:
            if strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff: #self.sims_score:
                return True
        return False

    def exclude_str_p(self, tokens_list : list, txt_list : list, coeff):
        length = len(tokens_list)
        txts_split = [txt_list[i:i+length] for i in range(0, len(txt_list), 1) if len(txt_list[i:i+length]) == length]
        for tx_l in txts_split:
            if strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff: #self.sims_score:
                return False
        return True

class SiameseNnDoc2VecClassifier(AbstractRules):
    def __init__(self, loader_obj):
        self.model_types = [("siamese_lstm_d2v", None)]
        self.model = loader_obj
        self.tknz = TokenizerApply(self.model)
        self.tkz_model = self.tknz.model_tokenize()
    
    def rules_apply(self, texts):
        session = keras.backend.get_session()
        init = tf.global_variables_initializer()
        session.run(init)
        text_vectors = self.tknz.txts_processing(texts)
        et_vectors = self.tkz_model.application_field["texts"]
        coeffs = self.tkz_model.application_field["coeff"]
        tags = self.tkz_model.application_field["tags"]

        decisions = []
        vcs_arr = np.array(et_vectors)
        
        global graph
        graph = tf.get_default_graph()
        
        for num, text_vector in enumerate(text_vectors):
            tx_tensor = np.array([text_vector for i in range(vcs_arr.shape[0])])
            tx_tensor = tx_tensor.reshape(vcs_arr.shape[0], vcs_arr.shape[1], 1)
            vcs_arr = vcs_arr.reshape(vcs_arr.shape[0], vcs_arr.shape[1], 1)
            with graph.as_default():
                scores = self.model.rules_algorithms["siamese_lstm_model"].predict([tx_tensor, vcs_arr])
            trues =  [(tg, True) for scr, cf, tg in zip(scores, coeffs, tags) if scr < cf]
            falses = [(tg, False) for scr, cf, tg in zip(scores, coeffs, tags) if scr > cf]
            decisions.append((num, trues+falses))

        return decisions


# Основное время тратится на загрузку лемматизатора и на лемматизацию эталонов
class ModelsChain(AbstractRules):
    def __init__(self, models):
        self.models = models
        self.classes = [SimpleRules, SiameseNnDoc2VecClassifier]
        self.classes_models = self.classes_modles_fill()

    def classes_modles_fill(self):
        classes_with_model = []
        for model in self.models:
            for Class in self.classes:
                try:
                    class_with_model = Class(model)
                    if model.model_type in [x[0] for x in class_with_model.model_types]:
                        #work_classes.append(class_with_model)
                        classes_with_model.append((class_with_model, model))
                except:
                    print("не удалось сопоставить классы с моделями", Class, model.model_type)
                    sys.exit(1)
        return classes_with_model
        
    # функция, применяющая набор моделей (цепочку моделей) к входящему тексту
    # допущение - модели должны содержать одинаковые эталоны с одинаковыми тегами
    # models :[] -> [loader_obj, ...]
    #true_tags = classes_models[0][1].application_field["tags"]
    def rules_apply(self, texts):    # выбор классов для полученных моделей:
        results = []
        all_tags = self.classes_models[0][1].application_field["tags"]
        for i in range(len(texts)):
            results.append(all_tags)

        for Class_with_model, model in self.classes_models:
            cls_results = Class_with_model.rules_apply(texts)
            true_rules_result = []
            for tx_result in cls_results:
                true_tags = [tx_res_tpl[0] for tx_res_tpl in tx_result[1] if tx_res_tpl[1] == True]                
                true_rules_result.append(true_tags)
            results = [intersection(x, y)  for x, y in zip(results, true_rules_result)]
        return results

        


if __name__ == "__main__":
    data_rout = r'./data'
    models_rout = r'./models'

    
    with open(os.path.join(models_rout, "include_and_model.pickle"), "br") as f:
        model1 = pickle.load(f)
    
    with open(os.path.join(models_rout, "intersec_share_model.pickle"), "br") as f:
        model2 = pickle.load(f)

    with open(os.path.join(models_rout, "siamese_lstm_d2v_bss.pickle"), "br") as f:
        model3 = pickle.load(f)

    with open(os.path.join(models_rout, "tax_demands_simple_model.pickle"), "br") as f:
        model4 = pickle.load(f)
    #tx = ["парит летит воздушный судно", "мой дядя самых чеСТных Правил ЮЛя новый год Фи ТАм котОРый", "солнце встает над рекой Хуанхе", 'упрощенная бухгалтерская отчетность кто сдает', "усн ип срок"]
    #tx = ["кто имеет право сдавать упрощенную бухгалтерскую отчетность", "кто сдает упрощенную отчетность", "зеленый крокодил"]

    tx = ["кто сдает упрощенную отчетность"]
    #rt_t = models_chain_apply([Loader(model3), Loader(model1), Loader(model2)], tx)

    """а теперь, если перейти от функции к классам (оценка времени):"""
    '''
    mdschain = ModelsChain([Loader(model1)])    
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print("model1:", rt_t, time.time() - t1)

    mdschain = ModelsChain([Loader(model2)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print("model2:", rt_t, time.time() - t1)

    mdschain = ModelsChain([Loader(model1), Loader(model2)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print("model1+model2:", rt_t, time.time() - t1)
    

    mdschain = ModelsChain([Loader(model3)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print("model3:", rt_t, time.time() - t1)

    mdschain = ModelsChain([Loader(model2), Loader(model3)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print("model2+model3:", rt_t, time.time() - t1)
    '''

    mdschain = ModelsChain([Loader(model4)])
    tx = ["в ходе проведения камеральной налоговой проверки на основе Расчет сумм налога на доходы физических лиц, исчисленных и удержанных налоговым агентом (форма № 6-НДФЛ) нулевая"]
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print("model4:", rt_t, time.time() - t1)


    '''
    mdschain = ModelsChain([Loader(model2), Loader(model3)])

    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)

    print("model2, model3:", rt_t, time.time() - t1)

    #print(SimpleTokenizer(Loader(model)).txts_processing(tx))
    #simplclass = SimpleRules(Loader(model))

    #print(simplclass.rules_apply(tx))
    '''