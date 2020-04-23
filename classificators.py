# здесь будут объекты для создания правил
# https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
# https://kobkrit.com/tensor-something-is-not-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1

import os, pickle, logging
import numpy as np
from abc import ABC, abstractmethod
from utility import Loader, intersection, strings_similarities  # contrastive_loss
from texts_processors import TokenizerApply
import tensorflow as tf
from itertools import groupby
from gensim.similarities import Similarity


class AbstractRules(ABC):
    def __init__(self):
        # перменная, описывающая, какие модели входят в класс
        self.model_types = []

    # должен возвращать структуру типа: [(num, [(tag, True), ...]), ...]
    # num - номер текста
    # tag - номер текста
    # True / False - результат для данного тега и данного текста
    @abstractmethod
    def rules_apply(self, text: []):
        pass


class SimpleRules(AbstractRules):
    def __init__(self, loader_obj):
        self.model_types = [("simple_rules", None)]
        self.functions_dict = {"include_and": self.include_and, "include_or": self.include_or,
                               "exclude_and": self.exclude_and, "exclude_or": self.exclude_or,
                               "include_str": self.include_str, "include_str_p": self.include_str_p,
                               "exclude_str_p": self.exclude_str_p, "intersec_share": self.intersec_share}
        self.model = loader_obj
        self.tokenizer = TokenizerApply(self.model)
        self.tknz_model = self.tokenizer.model_tokenize()

    def rules_apply(self, texts):
        decisions = []
        # применим правило к токенизированным текстам:
        for num, tknz_tx in enumerate(self.tokenizer.texts_processing(texts)):
            decisions_temp = []
            model_params = list(zip(self.tknz_model.application_field["tags"],
                                    self.tknz_model.application_field["rules"],
                                    self.tknz_model.application_field["texts"],
                                    self.tknz_model.application_field["coeff"]))
            # grouping rules with the same tag
            model_params_grouped = [(x, list(y)) for x, y in
                                    groupby(sorted(model_params, key=lambda x: x[0]), key=lambda x: x[0])]
            # оценка результатов применения правил для каждого тега (в каждой группе):
            for group, rules_list in model_params_grouped:
                decision = True
                for tg, rule, tknz_etalon, coeff in rules_list:
                    decision = decision and self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)
                decisions_temp.append((group, decision))
            decisions.append((num, decisions_temp))
        return decisions

    def include_and(self, tokens_list, text_list, coeff=0.0):
        for token in tokens_list:
            if token not in text_list:
                return False
        return True

    def include_or(self, tokens_list, text_list, coeff=0.0):
        for token in tokens_list:
            if token in text_list:
                return True
        return False

    def exclude_and(self, tokens_list, text_list, coeff=0.0):
        for token in tokens_list:
            if token in text_list:
                return False
        return True

    def exclude_or(self, tokens_list, text_list, coeff=0.0):
        for token in tokens_list:
            if token not in text_list:
                return True
        return False

    def intersec_share(self, tokens_list, text_list, intersec_coeff=0.7):
        intersec_tks = intersection(tokens_list, text_list)
        if len(intersec_tks) / len(tokens_list) > intersec_coeff:
            return True
        else:
            return False

    # функция, анализирующая на вхождение в текст строки (последовательности токенов, а не токенов по-отдельности)
    def include_str(self, tokens_str, text_str, coeff=0.0):
        if tokens_str in text_str:
            return True
        else:
            return False

    def exclude_str(self, tokens_str, text_str, coeff=0.0):
        if tokens_str not in text_str:
            return True
        else:
            return False

    def include_str_p(self, tokens_list: list, txt_list: list, coeff):
        length = len(tokens_list)
        txts_split = [txt_list[i:i + length] for i in range(0, len(txt_list), 1) if
                      len(txt_list[i:i + length]) == length]
        for tx_l in txts_split:
            if strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff:  # self.sims_score:
                return True
        return False

    def exclude_str_p(self, tokens_list: list, txt_list: list, coeff):
        length = len(tokens_list)
        txts_split = [txt_list[i:i + length] for i in range(0, len(txt_list), 1) if
                      len(txt_list[i:i + length]) == length]
        for tx_l in txts_split:
            if strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff:  # self.sims_score:
                return False
        return True


class SiameseNnDoc2VecClassifier(AbstractRules):
    def __init__(self, loader_obj):
        self.model_types = [("siamese_lstm_d2v", None)]
        self.model = loader_obj
        self.tknz = TokenizerApply(self.model)
        self.tkz_model = self.tknz.model_tokenize()

    def rules_apply(self, texts):
        text_vectors = self.tknz.texts_processing(texts)
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
                scores = self.model.classificator_algorithms["siamese_lstm_model"].predict([tx_tensor, vcs_arr])
            trues = [(tg, True) for scr, cf, tg in zip(scores, coeffs, tags) if scr < cf]
            falses = [(tg, False) for scr, cf, tg in zip(scores, coeffs, tags) if scr > cf]
            decisions.append((num, trues + falses))

        return decisions


class LsiClassifier(AbstractRules):
    def __init__(self, loader_obj):
        self.model_types = [("lsi", None)]
        self.model = loader_obj
        self.tknz = TokenizerApply(self.model)
        self.tkz_model = self.tknz.model_tokenize()
        self.et_vectors = self.tkz_model.application_field["texts"]
        self.coeffs = self.tkz_model.application_field["coeff"]
        self.tags = self.tkz_model.application_field["tags"]
        self.index = Similarity(None, self.et_vectors, num_features=self.model.texts_algorithms["num_topics"])

    def rules_apply(self, texts):
        text_vectors = self.tknz.texts_processing(texts)
        texts_tags_similarity = []
        for num, text_vector in enumerate(text_vectors):
            trues = [(tg, True) for tg, scr, cf in list(zip(self.tags, self.index[text_vector],
                                                            self.coeffs)) if scr > cf]
            falses = [(tg, False) for tg, scr, cf in list(zip(self.tags, self.index[text_vector], self.coeffs))
                      if scr < cf]
            texts_tags_similarity.append((num, trues + falses))
        return texts_tags_similarity


# Основное время тратится на загрузку лемматизатора и на лемматизацию эталонов
class ModelsChain(AbstractRules):
    def __init__(self, models):
        self.models = models
        self.classes = [SimpleRules, SiameseNnDoc2VecClassifier, LsiClassifier]
        self.classes_models = self.classes_modles_fill()

    def classes_modles_fill(self):
        classes_with_model = []
        for model in self.models:
            for Class in self.classes:
                try:
                    class_with_model = Class(model)
                    if model.model_type in [x[0] for x in class_with_model.model_types]:
                        classes_with_model.append((class_with_model, model))
                except Exception as exx:
                    logging.error('unable to link classes to models: "{}" to "{}"'.format(Class, model.model_type))
                    logging.error(exx)
        return classes_with_model

    # функция, применяющая набор моделей (цепочку моделей) к входящему тексту
    # допущение - модели должны содержать одинаковые эталоны с одинаковыми тегами
    # models :[] -> [loader_obj, ...]
    # true_tags = classes_models[0][1].application_field["tags"]
    def rules_apply(self, texts):  # выбор классов для полученных моделей:
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
            results = [intersection(x, y) for x, y in zip(results, true_rules_result)]
        return results


if __name__ == "__main__":
    import time

    data_rout = r'./data'
    models_rout = r'./models'

    with open(os.path.join(models_rout, "fast_answrs", "kosgu_lsi_model.pickle"), "br") as f:
        model = pickle.load(f)

    print(model["model_type"])
    for i in model:
        print(i)
    cl = LsiClassifier(Loader(model))
    mc = ModelsChain([Loader(model)])
    tx = "командировки статья косгу"
    t1 = time.time()
    print(mc.rules_apply([tx]), time.time()-t1)

"""
    with open(os.path.join(models_rout, "fast_answrs", "kosgu_incl_and_test_model.pickle"), "br") as f:
        kosgu_incl_and = pickle.load(f)

    with open(os.path.join(models_rout, "fast_answrs", "bss_lsi_model.pickle"), "br") as f:
        bss_lsi = pickle.load(f)

    with open(os.path.join(models_rout, "fast_answrs", "bss_intersec_share_model.pickle"), "br") as f:
        bss_intersec = pickle.load(f)

    with open(os.path.join(models_rout, "fast_answrs", "bss_include_and_model.pickle"), "br") as f:
        bss_include_and = pickle.load(f)

    with open(os.path.join(models_rout, "fast_answrs", "bss_siamese_lstm_d2v.pickle"), "br") as f:
        bss_siamese = pickle.load(f)

    tx = ["шпаргалка, чтобы определить квр и косгу для командировочных расходов госучреждений"]
    mdschain = ModelsChain([Loader(kosgu_incl_and)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "kosgu_incl_and:", rt_t, time.time() - t1)

    # tx = ["кто может применять упрощенный баланс"]
    tx = ["упрощенная финансовая отчетность кто сдает"]
    mdschain = ModelsChain([Loader(bss_lsi)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "bss_lsi:", rt_t, time.time() - t1)

    tx = ["кто может применять упрощенный баланс"]
    mdschain = ModelsChain([Loader(bss_intersec)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "bss_intersec:", rt_t, time.time() - t1)

    tx = ["кто может не применять ккт"]
    mdschain = ModelsChain([Loader(bss_include_and)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "bss_include_and:", rt_t, time.time() - t1)

    tx = ["кто может применять упрощенный баланс"]
    mdschain = ModelsChain([Loader(bss_siamese)])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "bss_siamese:", rt_t, time.time() - t1)
"""