from os.path import join
import pandas as pd
import numpy as np
import os
from src.knowledgegraph import KnowledgeGraph
import numpy as np
from src.utils import get_language_list, get_graph
import string
import time
import logging




class ParseData(object):
    '''
    Revised!
    '''
    def __init__(self, args, logger):
        self.data_path = args.data_path
        self.data_entity = self.data_path + "/entity/"
        self.data_kg = self.data_path + "/kg/"
        self.data_align_train = self.data_path + "/seed_train_pairs/"
        self.data_align_test = self.data_path + "/seed_test_pairs/"
        self.args = args

        self.target_kg = args.target_language
        self.kg_names = get_language_list(self.data_path, logger) # all kg names, sorted
        self.num_kgs = len(self.kg_names)

    def apply_noise(self, array, noise_pc):
        '''
        Revised!
        '''
        dim = array.shape[1]
        num_change_each = int(noise_pc * dim)
        if num_change_each == 0:
            return array

        mean = np.mean(array)
        std = np.std(array)

        for i in range(len(array)):
            to_change = np.random.randint(0, dim, num_change_each)
            value = np.random.normal(mean, std, num_change_each)
            array[i][to_change] = value 
        
        return array


    def load_data(self, noise_rate=0, logger=None):
        '''
        Revised!
        '''
        ent_name_path = '{}/ent_name_emb.npy'.format(self.data_path)
      
        logger.info('Loading embedding name from file...')
        entity_name_emb = np.load(ent_name_path)

        entity_name_emb = self.apply_noise(entity_name_emb, noise_rate)
        kg_object_dict, seeds_train, seeds_test = self.create_KG_objects_and_alignment() 
        self.num_relations = kg_object_dict[self.target_kg].num_relation * self.num_kgs 
        return kg_object_dict, seeds_train, seeds_test, entity_name_emb


    def load_all_to_all_seed_align_links(self):
        '''
        Revised!
        '''

        seeds_train = {}
        for f in os.listdir(self.data_align_train):  # e.g. 'el-en.tsv'
            lang1 = f[0:2]
            lang2 = f[3:5]
            links = pd.read_csv(join(self.data_align_train, f), sep='\t',header=None).values.astype(int)  # [N,2] ndarray
            seeds_train[(lang1, lang2)] = links # still be original index

        seeds_test = {}
        for f in os.listdir(self.data_align_test):  # e.g. 'el-en.tsv'
            lang1 = f[0:2]
            lang2 = f[3:5]
            links = pd.read_csv(join(self.data_align_test, f), sep='\t',header=None).values.astype(int)  # [N,2] ndarray
            seeds_test[(lang1, lang2)] = links # still be original index
        return seeds_train, seeds_test # dict of numpy array!!


    def create_KG_objects_and_alignment(self):
        '''
        Revised!
        '''
        entity_base = 0
        relation_base = 0
        kg_objects_dict = {} 

        for lang in self.kg_names:
            kg_train_data, kg_val_data, kg_test_data, entity_num, relation_num = self.load_kg_data(lang) 

            if lang == self.target_kg:
                is_supporter_kg = False
            else:
                is_supporter_kg = True

            kg_each = KnowledgeGraph(lang, kg_train_data, kg_val_data, kg_test_data, entity_num, relation_num, is_supporter_kg,
                                     entity_base, relation_base, self.args.device)

            entity_base += entity_num            
            relation_base += relation_num
            kg_each.upper_entity_base = entity_base
            kg_each.upper_relation_base = relation_base
            kg_objects_dict[lang] = kg_each

        self.num_entities = entity_base

        for lang in self.kg_names:
            if lang == self.target_kg:
                is_target_KG = True
            else:
                is_target_KG = False
            kg_lang = kg_objects_dict[lang]
            edge_index, edge_type =  get_graph(self.data_path, lang, is_target_KG)
            kg_lang.edge_index = edge_index # numpy array
            kg_lang.edge_type = edge_type

        seeds_train, seeds_test = self.load_all_to_all_seed_align_links() # None, links, None # not include base index yet!

        return kg_objects_dict, seeds_train, seeds_test


    def load_kg_data(self, language):
        '''
        Revised!
        '''
        train_df = pd.read_csv(join(self.data_kg, language + '-train.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])
        val_df = pd.read_csv(join(self.data_kg, language + '-val.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])
        test_df = pd.read_csv(join(self.data_kg, language + '-test.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])

        f = open(self.data_entity + language + '.tsv','r',encoding='utf-8')
        lines = f.readlines()
        f.close()

        entity_num = len(lines)

        relation_list = [line.rstrip() for line in open(join(self.data_path, 'relations.txt'))]
        relation_num = len(relation_list) + 1

        triples_train = train_df.values.astype(np.int)
        triples_val = val_df.values.astype(np.int)
        triples_test = test_df.values.astype(np.int)

        return triples_train, triples_val, triples_test, entity_num, relation_num

