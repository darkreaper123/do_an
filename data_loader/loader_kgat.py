import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import *


class DataLoaderKGAT(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        print(1)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        self.test_cold_batch_size = args.test_batch_size
        self.test_warm_batch_size = args.test_batch_size
        print(2)
        kg_data = self.load_kg(self.kg_file)
        print(3)
        self.construct_data(kg_data)
        print(4)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def contruct_new_data(self):
        cf2kg_train_data = pd.DataFrame(np.zeros((len(self.cf_val_data[0]), 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_val_data[0]
        cf2kg_train_data['t'] = self.cf_val_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((len(self.cf_val_data[1]), 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_val_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_val_data[0]

        self.kg_train_data = pd.concat([self.kg_train_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)
        self.h_list = torch.LongTensor(self.kg_train_data["h"].values)
        self.t_list = torch.LongTensor(self.kg_train_data["t"].values)
        self.r_list = torch.LongTensor(self.kg_train_data["r"].values)
    def new_init(self):
        self.contruct_new_data()
        self.create_adjacency_dict()
        self.create_laplacian_dict()
    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)
        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r'].astype(int)) + 1
        self.n_entities = max(max(kg_data['h'].astype(int)), max(kg_data['t'].astype(int))) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))
        self.cf_val_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_val_data[0]))).astype(np.int32), self.cf_val_data[1].astype(np.int32))


        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}
        self.val_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.val_user_dict.items()}
        self.test_cold_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_cold_user_dict.items()}
        self.test_warm_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_warm_user_dict.items()}
        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        #self.train_kg_dict = collections.defaultdict(list)
        #self.train_relation_dict = collections.defaultdict(list)
        def get_dict_one_two_targets(source_column, two_target_columns):
           self.kg_train_data_new = self.kg_train_data.copy()
           for column in two_target_columns:
               try:
                   self.kg_train_data_new["two_columns"] += self.kg_train_data_new[column].apply(lambda x : [x]) 
               except:
                   self.kg_train_data_new["two_columns"] = self.kg_train_data_new[column].apply(lambda x : [x]) 
           self.kg_train_data_new["two_columns"] = self.kg_train_data_new["two_columns"].apply(lambda x : tuple(x))
           self.kg_train_data_new = self.kg_train_data_new.groupby(source_column, as_index = False).agg(lambda x : [elem for elem in x])
           return collections.defaultdict(list, dict(self.kg_train_data_new[[source_column, "two_columns"]].values))
        self.train_kg_dict = get_dict_one_two_targets("h", ["t","r"])
        self.train_relation_dict = get_dict_one_two_targets("r", ["h","t"])
        self.h_list = torch.LongTensor(self.kg_train_data["h"].values)
        self.t_list = torch.LongTensor(self.kg_train_data["t"].values)
        self.r_list = torch.LongTensor(self.kg_train_data["r"].values)

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            #print(rowsum)
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)


class DataLoaderKGAT_Replay(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        print(1)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        self.test_cold_batch_size = args.test_batch_size
        self.test_val_batch_size = args.test_batch_size
        self.test_warm_batch_size = args.test_batch_size
        print(2)
        kg_data = self.load_kg(self.kg_file)
        print(3)
        self.construct_data(kg_data)
        print(4)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        only_me_head_kg = pd.DataFrame({'h': kg_data['h'].unique(), 
                                        'r': np.zeros((kg_data['h'].nunique(),)).astype(int), 
                                        't': kg_data['h'].unique()})
        only_me_tail_kg = pd.DataFrame({'h': kg_data['t'].unique(), 
                                        'r': (np.zeros((kg_data['t'].nunique(),)) + n_relations).astype(int), 
                                        't': kg_data['t'].unique()})
        kg_data = pd.concat([kg_data, inverse_kg_data, only_me_head_kg, only_me_tail_kg], axis=0, ignore_index=True, sort=False)
        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}
        self.test_cold_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_cold_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        #self.train_kg_dict = collections.defaultdict(list)
        #self.train_relation_dict = collections.defaultdict(list)
        def get_dict_one_two_targets(source_column, two_target_columns):
           self.kg_train_data_new = self.kg_train_data.copy()
           for column in two_target_columns:
               try:
                   self.kg_train_data_new["two_columns"] += self.kg_train_data_new[column].apply(lambda x : [x]) 
               except:
                   self.kg_train_data_new["two_columns"] = self.kg_train_data_new[column].apply(lambda x : [x]) 
           self.kg_train_data_new["two_columns"] = self.kg_train_data_new["two_columns"].apply(lambda x : tuple(x))
           self.kg_train_data_new = self.kg_train_data_new.groupby(source_column, as_index = False).agg(lambda x : [elem for elem in x])
           return collections.defaultdict(list, dict(self.kg_train_data_new[[source_column, "two_columns"]].values))
        self.train_kg_dict = get_dict_one_two_targets("h", ["t","r"])
        self.train_relation_dict = get_dict_one_two_targets("r", ["h","t"])
        self.h_list = torch.LongTensor(self.kg_train_data["h"].values)
        self.t_list = torch.LongTensor(self.kg_train_data["t"].values)
        self.r_list = torch.LongTensor(self.kg_train_data["r"].values)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            #print(rowsum)
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)
        
        
        
        
        
        
class DataLoaderKGAT_MultiTask_full(DataLoaderBase_MultiTask_full):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        print(1)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        self.test_cold_batch_size = args.test_batch_size
        print(2)
        kg_data = self.load_kg(self.kg_file)
        print(3)
        self.construct_data(kg_data)
        print(4)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)
        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        if self.n_entities < self.n_items:
            self.n_entities = self.n_items
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}
        self.test_cold_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_cold_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        #self.train_kg_dict = collections.defaultdict(list)
        #self.train_relation_dict = collections.defaultdict(list)
        def get_dict_one_two_targets(source_column, two_target_columns):
           self.kg_train_data_new = self.kg_train_data.copy()
           for column in two_target_columns:
               try:
                   self.kg_train_data_new["two_columns"] += self.kg_train_data_new[column].apply(lambda x : [x]) 
               except:
                   self.kg_train_data_new["two_columns"] = self.kg_train_data_new[column].apply(lambda x : [x]) 
           self.kg_train_data_new["two_columns"] = self.kg_train_data_new["two_columns"].apply(lambda x : tuple(x))
           self.kg_train_data_new = self.kg_train_data_new.groupby(source_column, as_index = False).agg(lambda x : [elem for elem in x])
           return collections.defaultdict(list, dict(self.kg_train_data_new[[source_column, "two_columns"]].values))
        self.train_kg_dict = get_dict_one_two_targets("h", ["t","r"])
        self.train_relation_dict = get_dict_one_two_targets("r", ["h","t"])
        self.h_list = torch.LongTensor(self.kg_train_data["h"].values)
        self.t_list = torch.LongTensor(self.kg_train_data["t"].values)
        self.r_list = torch.LongTensor(self.kg_train_data["r"].values)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            #print(rowsum)
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)
        
        
class DataLoaderKGAT_MultiTask(DataLoaderBase_MultiTask):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        print(1)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        self.test_cold_batch_size = args.test_batch_size
        print(2)
        kg_data = self.load_kg(self.kg_file)
        print(3)
        self.construct_data(kg_data)
        print(4)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)
        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}
        self.test_cold_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_cold_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        #self.train_kg_dict = collections.defaultdict(list)
        #self.train_relation_dict = collections.defaultdict(list)
        def get_dict_one_two_targets(source_column, two_target_columns):
           self.kg_train_data_new = self.kg_train_data.copy()
           for column in two_target_columns:
               try:
                   self.kg_train_data_new["two_columns"] += self.kg_train_data_new[column].apply(lambda x : [x]) 
               except:
                   self.kg_train_data_new["two_columns"] = self.kg_train_data_new[column].apply(lambda x : [x]) 
           self.kg_train_data_new["two_columns"] = self.kg_train_data_new["two_columns"].apply(lambda x : tuple(x))
           self.kg_train_data_new = self.kg_train_data_new.groupby(source_column, as_index = False).agg(lambda x : [elem for elem in x])
           return collections.defaultdict(list, dict(self.kg_train_data_new[[source_column, "two_columns"]].values))
        self.train_kg_dict = get_dict_one_two_targets("h", ["t","r"])
        self.train_relation_dict = get_dict_one_two_targets("r", ["h","t"])
        self.h_list = torch.LongTensor(self.kg_train_data["h"].values)
        self.t_list = torch.LongTensor(self.kg_train_data["t"].values)
        self.r_list = torch.LongTensor(self.kg_train_data["r"].values)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            #print(rowsum)
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)
        
class DataLoaderKGAT_MultiTask_full(DataLoaderBase_MultiTask_full):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        print(1)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        self.test_cold_batch_size = args.test_batch_size
        print(2)
        kg_data = self.load_kg(self.kg_file)
        print(3)
        self.construct_data(kg_data)
        print(4)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)
        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        if self.n_entities < self.n_items:
            self.n_entities = self.n_items
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}
        self.test_cold_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_cold_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        #self.train_kg_dict = collections.defaultdict(list)
        #self.train_relation_dict = collections.defaultdict(list)
        def get_dict_one_two_targets(source_column, two_target_columns):
           self.kg_train_data_new = self.kg_train_data.copy()
           for column in two_target_columns:
               try:
                   self.kg_train_data_new["two_columns"] += self.kg_train_data_new[column].apply(lambda x : [x]) 
               except:
                   self.kg_train_data_new["two_columns"] = self.kg_train_data_new[column].apply(lambda x : [x]) 
           self.kg_train_data_new["two_columns"] = self.kg_train_data_new["two_columns"].apply(lambda x : tuple(x))
           self.kg_train_data_new = self.kg_train_data_new.groupby(source_column, as_index = False).agg(lambda x : [elem for elem in x])
           return collections.defaultdict(list, dict(self.kg_train_data_new[[source_column, "two_columns"]].values))
        self.train_kg_dict = get_dict_one_two_targets("h", ["t","r"])
        self.train_relation_dict = get_dict_one_two_targets("r", ["h","t"])
        self.h_list = torch.LongTensor(self.kg_train_data["h"].values)
        self.t_list = torch.LongTensor(self.kg_train_data["t"].values)
        self.r_list = torch.LongTensor(self.kg_train_data["r"].values)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            #print(rowsum)
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)
        
        
class DataLoaderKGAT_MultiTask_full_2(DataLoaderBase_MultiTask_full_2):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        print(1)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        self.test_cold_batch_size = args.test_batch_size
        print(2)
        kg_data = self.load_kg(self.kg_file)
        print(3)
        self.construct_data(kg_data)
        print(4)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)
        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        #self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_entities = self.n_items
        self.n_users_entities = self.n_users + self.n_entities
        print(self.n_users_entities)
        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        #self.cf_val_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_val_data[0]))).astype(np.int32), self.cf_val_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        #self.val_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.val_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}
        self.test_cold_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_cold_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        #self.train_kg_dict = collections.defaultdict(list)
        #self.train_relation_dict = collections.defaultdict(list)
        def get_dict_one_two_targets(source_column, two_target_columns):
           self.kg_train_data_new = self.kg_train_data.copy()
           for column in two_target_columns:
               try:
                   self.kg_train_data_new["two_columns"] += self.kg_train_data_new[column].apply(lambda x : [x]) 
               except:
                   self.kg_train_data_new["two_columns"] = self.kg_train_data_new[column].apply(lambda x : [x]) 
           self.kg_train_data_new["two_columns"] = self.kg_train_data_new["two_columns"].apply(lambda x : tuple(x))
           self.kg_train_data_new = self.kg_train_data_new.groupby(source_column, as_index = False).agg(lambda x : [elem for elem in x])
           return collections.defaultdict(list, dict(self.kg_train_data_new[[source_column, "two_columns"]].values))
        self.train_kg_dict = get_dict_one_two_targets("h", ["t","r"])
        self.train_relation_dict = get_dict_one_two_targets("r", ["h","t"])
        self.h_list = torch.LongTensor(self.kg_train_data["h"].values)
        self.t_list = torch.LongTensor(self.kg_train_data["t"].values)
        self.r_list = torch.LongTensor(self.kg_train_data["r"].values)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            #print(rowsum)
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)