import os
import time
import random
import collections

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
class DataLoaderBase(object):

    def __init__(self, args, logging):
        self.n_interactions_max = 1
        self.full_sampling = args.full_sampling
        self.args = args
        self.n_interactions = args.n_interactions
        self.data_centric = args.data_centric
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.test_cold_file = os.path.join(self.data_dir, 'test_cold.txt')
        self.test_warm_file = os.path.join(self.data_dir, 'test_warm.txt')
        self.val_file = os.path.join(self.data_dir, 'val.txt')
        self.train_data_masking_test_file = os.path.join(self.data_dir, 'train_for_test.txt')
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        try:
            self.user_features = pd.read_csv(os.path.join(self.data_dir, "user_features.csv"))
        except:
            self.user_features = None
        try:
            self.item_features = pd.read_csv(os.path.join(self.data_dir, "item_features.csv"))
        except:
            self.item_features = None   
        print(self.item_features)
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
        self.cf_test_cold_data, self.test_cold_user_dict = self.load_cf(self.test_cold_file)
        self.cf_test_warm_data, self.test_warm_user_dict = self.load_cf(self.test_warm_file)
        self.cf_val_data, self.val_user_dict = self.load_cf(self.val_file)
        _, self.train_data_masking_test_user_dict = self.load_cf(self.train_data_masking_test_file)
        self.statistic_cf() 
        self.get_items_unpopularity(self.train_file)
        #self.user_category_columns = []
        try:
            self.user_numeric_columns = open(os.path.join(self.data_dir, 'user_numeric_name_columns.txt'), 'r').read().split('\t')
        except:
            self.user_numeric_columns = ['']
        if self.user_numeric_columns[0] == '':
          self.user_numeric_columns = []
        try:
            self.user_category_columns = open(os.path.join(self.data_dir, 'user_category_name_columns.txt'), 'r').read().split('\t')
        except:
            self.user_category_columns = ['']
        if self.user_category_columns[0] == '':
          self.user_category_columns = []
        #self.item_category_columns = ["all_categories"]
        try:
            self.item_category_columns = open(os.path.join(self.data_dir, 'item_category_name_columns.txt'), 'r').read().split('\t')
        except:
            self.item_category_columns = ['']
        if self.item_category_columns[0] == '':
          self.item_category_columns = []
        #self.item_numeric_columns = ["duration", "imdb_rating", "release_year"]
        try:
            self.item_numeric_columns = open(os.path.join(self.data_dir, 'item_numeric_name_columns.txt'), 'r').read().split('\t')
        except:
            self.item_numeric_columns = ['']
        if self.item_numeric_columns[0] == '':
          self.item_numeric_columns = []
        self.user_n_interactions = None
        if self.data_centric == 1.:
            scale = 0.5
            self.user_n_interactions = self.user_features["user"].apply(lambda x : (len(self.train_user_dict[int(x)]) * scale if int(x) in self.train_user_dict else 0)).values
        def category_to_id(data, column):
            to_new_id = dict(np.hstack((data[column][pd.isnull(data[column]) == False].unique().reshape(-1, 1),
                                        np.linspace(2, data[column].nunique() + 1, data[column].nunique()).astype(int).reshape(-1, 1))))
            data[column][pd.isnull(data[column]) == False] = data[column][pd.isnull(data[column]) == False].apply(lambda x : to_new_id[x])
            data[column][pd.isnull(data[column])] = 1
            return data, to_new_id
        all_categorys_to_id = {}
        for column in self.user_category_columns:
            if self.user_features is None:
                break
            self.user_features, new_category_to_id = category_to_id(self.user_features, column)
            all_categorys_to_id[column] = new_category_to_id
        self.max_size_embedding_category = []
        self.max_len_category = []
        self.n_category = []
        self.n_numeric = [len(self.user_numeric_columns), len(self.item_numeric_columns)]
        def preprocess_profiles_user(recent_profiles, category_columns, numeric_columns, max_size_embedding_category, start_index = 3, max_len_category = None):
            new_max_size_category = max_size_embedding_category
            profiles = recent_profiles.copy()
            try:
                scaler = StandardScaler() 
                profiles[numeric_columns] = scaler.fit_transform(profiles[numeric_columns])
            except:
                print("Numeric features are not existed")
            columns_to_numeric_id = dict([(column, {}) for column in category_columns])
            for i, column in enumerate(category_columns):
                n_distinct_val = profiles[column].nunique()
                columns_to_numeric_id[column] = dict(np.hstack((profiles[column].unique().reshape(-1, 1), 
                                                                (np.linspace(0, n_distinct_val - 1, n_distinct_val) * max_len_category + i + start_index).reshape(-1, 1).astype(int))))
                new_max_size_category = max(new_max_size_category, int((n_distinct_val - 1) * max_len_category + i + start_index))
                columns_to_numeric_id[column][0] = 0
                profiles[column] = profiles[column].apply(lambda x : columns_to_numeric_id[column][x])
            return profiles, new_max_size_category
        def preprocess_profiles_item(recent_profiles, category_columns, numeric_columns):
            profiles = recent_profiles.copy()
            try:
                scaler = StandardScaler() 
                profiles[numeric_columns] = scaler.fit_transform(profiles[numeric_columns])
            except:
                print("Numeric features are not existed")
            columns_to_numeric_id = dict([(column, {}) for column in category_columns])
            profiles_categories = profiles[category_columns[0]].apply(lambda x : [int(elem) for elem in str(x).split("\t")])
            max_n_categories = int(profiles_categories.apply(lambda x : len(x)).max())
            profiles_categories = pad_sequences(profiles_categories.values, max_n_categories)
            new_max_size_category = np.max(profiles_categories.reshape(-1))
            return profiles, new_max_size_category, profiles_categories, max_n_categories 
        if self.use_pretrain == 1:
            self.load_pretrained_data()
        if self.user_features is not None:
            self.users_profiles, max_size_users_embedding_category, users_profiles_category, max_n_user_categories =  preprocess_profiles_item(self.user_features, self.user_category_columns, self.user_numeric_columns)
        if len(self.item_category_columns) > 0 or len(self.item_numeric_columns) > 0:
            self.items_profiles, max_size_items_embedding_category, items_profiles_category, max_n_item_categories =  preprocess_profiles_item(self.item_features, self.item_category_columns, self.item_numeric_columns)
        if self.user_features is not None or self.item_features is not None:
            try:
              self.max_len_category.append(max_n_user_categories)
              self.n_category.append(max_n_user_categories)
              self.max_size_embedding_category.append(max_size_users_embedding_category)
            except:
              self.max_len_category.append(0)
              self.n_category.append(0)
              self.max_size_embedding_category.append(0)
            try:
              self.max_len_category.append(max_n_item_categories)
              self.n_category.append(max_n_item_categories)
              self.max_size_embedding_category.append(max_size_items_embedding_category)
            except:
              self.max_len_category.append(0)
              self.n_category.append(0)
              self.max_size_embedding_category.append(0)
        print(self.max_size_embedding_category)      
        self.user_category_features = None
        self.user_numeric_features = None
        self.item_category_features = None
        self.item_numeric_features = None
        try:
            try:
                self.user_category_features = np.zeros((int(self.users_profiles["user"].max() + 1), max_n_user_categories))
                self.user_category_features[self.users_profiles["user"].values.astype(int)] = users_profiles_category
            except:
                print('User category feature is not existed')
            if len(self.user_numeric_columns):
                self.user_numeric_features = np.zeros((int(self.users_profiles["user"].max() + 1), len(self.user_numeric_columns)))
                self.user_numeric_features[self.users_profiles["user"].values.astype(int)] = self.users_profiles[self.user_numeric_columns]
            else:
                print('User numeric feature is not existed')
        except:
            print("User features is not existed")
        try:
            try:
                self.item_category_features = np.zeros((int(self.items_profiles["item"].max() + 1), max_n_item_categories))
                self.item_category_features[self.items_profiles["item"].values.astype(int)] = items_profiles_category
            except:
                print('Item category feature is not existed')
            if len(self.item_numeric_columns) > 0:
                self.item_numeric_features = np.zeros((int(self.items_profiles["item"].max() + 1), len(self.item_numeric_columns)))
                self.item_numeric_features[self.items_profiles["item"].values.astype(int)] = self.items_profiles[self.item_numeric_columns]
            else:
                print('Item category feature is not existed')
        except:
            print("Item features is not existed")
    def load_cf(self, filename):
        lines = pd.read_csv(filename, sep = "|", names = ["user", "items"])
        if self.n_interactions > 0:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")][::-1][:self.n_interactions])
        else:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
        lines["user"] = lines["user"].apply(lambda x : int(x))
        user_dict = dict(lines.values)
        if self.full_sampling:
            self.n_interactions_max = max(int(lines["items"].apply(lambda x : len(x)).max()), self.n_interactions_max)
        lines["items"] =  lines["items"].apply(lambda x : " ".join([str(elem) for elem in x]))
        lines["user"] = lines["user"].apply(lambda x : str(x))
        lines["user"] = lines["user"] + " " + lines["items"].apply(lambda x : str(len(x.split(" "))))
        lines["user"] = lines["user"].apply(lambda x : " ".join(list(np.full((int(x.split(" ")[1]),), x.split(" ")[0]))))
        user = np.array(" ".join(lines["user"].values).split(" "), dtype=np.int32)
        item = np.array(" ".join(lines["items"].values).split(" "), dtype=np.int32)
        return (user, item), user_dict
    def get_items_unpopularity(self, filename):
        lines = pd.read_csv(filename, sep = "|", names = ["user", "item"])
        lines["item"] =  lines["item"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
        lines = lines.explode("item")
        lines = lines.groupby("item", as_index = False).agg({'user':(lambda x : len(x))})
        lines = dict(lines[["user", "item"]].values)
        self.items_unpopularity = np.zeros((self.n_items,))
        self.items_unpopularity[list(lines.keys())] = np.asarray(list(lines.values()))
        self.items_unpopularity = 1 - (self.items_unpopularity - np.min(self.items_unpopularity))/(np.max(self.items_unpopularity) - np.min(self.items_unpopularity))
        


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0]))
        try:
          self.n_users = max(self.n_users, max(self.cf_val_data[0])) + 1
        except:
          print('No val')
        try:
          self.n_users = max(self.n_users, max(self.cf_test_cold_data[0])) + 1
        except:
          print('No test cold')
        try:
          self.n_users = max(self.n_users, max(self.cf_test_warm_data[0])) + 1
        except:
          print('No test warm')
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1]))
        try:
          self.n_items = max(self.n_items, max(self.cf_val_data[1])) + 1
        except:
          print('No val')
        try:
          self.n_items = max(self.n_items, max(self.cf_test_cold_data[1])) + 1
        except:
          print('No test cold')
        try:
          self.n_items = max(self.n_items, max(self.cf_test_warm_data[1])) + 1
        except:
          print('No test warm')
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        self.max_id = [max(self.cf_train_data[0]), max(self.cf_train_data[1])]
        for i in range(2):
          self.max_id[i] = max(self.max_id[i], max(self.cf_val_data[i]))
        for i in range(2):
          self.max_id[i] = max(self.max_id[i], max(self.cf_test_data[i]))
        for i in range(2):
          self.max_id[i] = max(self.max_id[i], max(self.cf_test_warm_data[i]))
          print('Max id {} : {}'.format('user' if i == 0 else 'item', self.max_id[i]))


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep = ' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data


    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)
        padding_zeros = []
        if n_sample_pos_items > n_pos_items:
            padding_zeros = np.zeros((n_sample_pos_items - n_pos_items,)) - 1
            padding_zeros = list(padding_zeros)
            n_sample_pos_items = n_pos_items
        random_item = np.random.permutation(n_pos_items)[:n_sample_pos_items]
        sample_pos_items = np.asarray(pos_items)[random_item] 
        return padding_zeros + list(sample_pos_items)
    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]
        sample_neg_items = []
        padding_zeros = []
        if n_sample_neg_items > len(pos_items):
            padding_zeros = np.zeros((n_sample_neg_items - len(pos_items),)) - 1
            padding_zeros = list(padding_zeros)
            n_sample_neg_items = len(pos_items)
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=1, high=self.n_items, size=1)[0]
            if self.full_sampling:
                if neg_item_id not in pos_items:
                    sample_neg_items.append(neg_item_id)
                continue
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return padding_zeros + sample_neg_items
    def sample_user(self, user_dict, user_id, n_sample):
        pos_items = user_dict[user_id]
        padding_zeros = []
        if n_sample > len(pos_items):
            padding_zeros = np.zeros((n_sample - len(pos_items),)) - 1
            padding_zeros = list(padding_zeros)
            n_sample = len(pos_items)
        return padding_zeros + list(np.full((n_sample,), user_id))
    def generate_cf_batch(self, user_dict, batch_size, reranking_loss = False):
        exist_users = list(user_dict.keys())
        #print(len(exist_users))
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        #n_pos = 5
        n_pos = 5
        if self.full_sampling:
            n_pos = self.n_interactions_max
        if reranking_loss and self.args.using_val_for_training_reranking_loss:
            n_pos = self.args.n_items_used_each_user_phase_2
            
        n_neg = n_pos
        n_users = n_pos
        batch_pos_item = [self.sample_pos_items_for_u(user_dict, u, n_pos) for u in batch_user]
        batch_pos_item = np.asarray(batch_pos_item).reshape(-1)
        batch_pos_item = batch_pos_item[batch_pos_item >= 0]
        batch_pos_item = torch.LongTensor(batch_pos_item)
        
        batch_user_2 = [self.sample_user(user_dict, u, n_users) for u in batch_user]
        batch_user_2 = np.asarray(batch_user_2)
        batch_user_2 = batch_user_2[batch_user_2 >= 0]
        batch_user_2 = torch.LongTensor(batch_user_2)
        
        if reranking_loss:
            return batch_user_2, batch_pos_item
        
        batch_neg_item = [self.sample_neg_items_for_u(user_dict, u, n_neg) for u in batch_user]
        batch_neg_item = np.asarray(batch_neg_item).reshape(-1)
        batch_neg_item = batch_neg_item[batch_neg_item >= 0]
        batch_neg_item = torch.LongTensor(batch_neg_item)
        
        return batch_user_2, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
        batch_relation_tail = [self.sample_pos_triples_for_h(kg_dict, h, 1)[0] +
                               self.sample_pos_triples_for_h(kg_dict, h, 1)[1] +
                               self.sample_neg_triples_for_h(kg_dict, h, self.sample_pos_triples_for_h(kg_dict, h, 1)[0], 1, highest_neg_idx)
                               for h in batch_head]
        batch_relation_tail = np.asarray(batch_relation_tail)
        batch_relation = batch_relation_tail[:, 0]
        batch_pos_tail = batch_relation_tail[:, 1]
        batch_neg_tail = batch_relation_tail[:, 2]
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embed_dim
        assert self.item_pre_embed.shape[1] == self.args.embed_dim
        
        
        
        
        
        
        
        
        
        
class DataLoaderBase_MultiTask(object):

    def __init__(self, args, logging):
        self.n_interactions_max = 1
        self.pretrain_task_id = args.pretrain_task_id
        self.full_sampling = args.full_sampling
        self.args = args
        self.n_interactions = args.n_interactions
        self.data_centric = args.data_centric
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.test_cold_file = os.path.join(self.data_dir, 'test_cold.txt')
        self.item_to_task_id = pd.read_csv(os.path.join(self.data_dir, "item_to_task_id.csv"))
        self.item_to_task_id["new_id"] = self.item_to_task_id["new_id"].astype(int)
        self.item_to_task_id["task_id"] = self.item_to_task_id["task_id"].astype(int)
        self.train_data_masking_test_file = os.path.join(self.data_dir, 'train_for_test.txt')
        try:
            self.user_features = pd.read_csv(os.path.join(self.data_dir, "user_features.csv"))
        except:
            self.user_features = None
        try:
            self.item_features = pd.read_csv(os.path.join(self.data_dir, "item_features.csv"))
        except:
            self.item_features = None   
        print(self.item_features)
        #self.load_cf_basic_full()
        self.load_full_cf(args.pretrain_task_id)
        self.statistic_cf()
        print("Max num interactions is {}".format(self.n_interactions_max))
        self.user_category_columns = ["gender", "province_name"]
        #self.user_numeric_columns = ["age", "n_profiles"]
        self.user_numeric_columns = []
        self.item_category_columns = ["all_categories"]
        #self.item_numeric_columns = ["duration", "imdb_rating", "release_year"]
        self.item_numeric_columns = []
        self.user_n_interactions = None
        if self.data_centric == 1.:
            scale = 0.5
            self.user_n_interactions = self.user_features["user"].apply(lambda x : (len(self.train_user_dict[int(x)]) * scale if int(x) in self.train_user_dict else 0)).values
        def category_to_id(data, column):
            to_new_id = dict(np.hstack((data[column][pd.isnull(data[column]) == False].unique().reshape(-1, 1),
                                        np.linspace(2, data[column].nunique() + 1, data[column].nunique()).astype(int).reshape(-1, 1))))
            data[column][pd.isnull(data[column]) == False] = data[column][pd.isnull(data[column]) == False].apply(lambda x : to_new_id[x])
            data[column][pd.isnull(data[column])] = 1
            return data, to_new_id
        all_categorys_to_id = {}
        for column in self.user_category_columns:
            if self.user_features is None:
                break
            self.user_features, new_category_to_id = category_to_id(self.user_features, column)
            all_categorys_to_id[column] = new_category_to_id
        self.max_size_embedding_category = [0]
        self.max_len_category = [len(self.user_category_columns)]
        self.n_category = [len(self.user_category_columns)]
        self.n_numeric = [len(self.user_numeric_columns), len(self.item_numeric_columns)]
        def preprocess_profiles_user(recent_profiles, category_columns, numeric_columns, max_size_embedding_category, start_index = 3, max_len_category = None):
            new_max_size_category = max_size_embedding_category
            profiles = recent_profiles.copy()
            try:
                scaler = StandardScaler() 
                profiles[numeric_columns] = scaler.fit_transform(profiles[numeric_columns])
            except:
                print("Numeric features are not existed")
            columns_to_numeric_id = dict([(column, {}) for column in category_columns])
            for i, column in enumerate(category_columns):
                n_distinct_val = profiles[column].nunique()
                columns_to_numeric_id[column] = dict(np.hstack((profiles[column].unique().reshape(-1, 1), 
                                                                (np.linspace(0, n_distinct_val - 1, n_distinct_val) * max_len_category + i + start_index).reshape(-1, 1).astype(int))))
                new_max_size_category = max(new_max_size_category, int((n_distinct_val - 1) * max_len_category + i + start_index))
                columns_to_numeric_id[column][0] = 0
                profiles[column] = profiles[column].apply(lambda x : columns_to_numeric_id[column][x])
            return profiles, new_max_size_category
        def preprocess_profiles_item(recent_profiles, category_columns, numeric_columns):
            profiles = recent_profiles.copy()
            try:
                scaler = StandardScaler() 
                profiles[numeric_columns] = scaler.fit_transform(profiles[numeric_columns])
            except:
                print("Numeric features are not existed")
            columns_to_numeric_id = dict([(column, {}) for column in category_columns])
            profiles_categories = profiles[category_columns[0]].apply(lambda x : [int(elem) for elem in str(x).split("\t")])
            max_n_categories = int(profiles_categories.apply(lambda x : len(x)).max())
            profiles_categories = pad_sequences(profiles_categories.values, max_n_categories)
            new_max_size_category = np.max(profiles_categories.reshape(-1))
            return profiles, new_max_size_category, profiles_categories, max_n_categories 
        if self.use_pretrain == 1:
            self.load_pretrained_data()
        if self.user_features is not None:
            self.users_profiles, self.max_size_embedding_category[0] =  preprocess_profiles_user(self.user_features, self.user_category_columns, self.user_numeric_columns, self.max_size_embedding_category[0], 3, self.max_len_category[0])
        if self.item_features is not None:
            self.items_profiles, max_size_items_embedding_category, items_profiles_category, max_n_categories =  preprocess_profiles_item(self.item_features, self.item_category_columns, self.item_numeric_columns)
        if self.user_features is not None or self.item_features is not None:
            self.max_len_category.append(max_n_categories)
            self.n_category.append(max_n_categories)
            self.max_size_embedding_category.append(max_size_items_embedding_category)
        self.user_category_features = None
        self.user_numeric_features = None
        self.item_category_features = None
        self.item_numeric_features = None
        try:
            self.user_category_features = np.zeros((int(self.users_profiles["user"].max() + 1), len(self.user_category_columns)))
            self.user_numeric_features = np.zeros((int(self.users_profiles["user"].max() + 1), len(self.user_numeric_columns)))
            self.user_category_features[self.users_profiles["user"].values.astype(int)] = self.users_profiles[self.user_category_columns]
            self.user_numeric_features[self.users_profiles["user"].values.astype(int)] = self.users_profiles[self.user_numeric_columns]
        except:
            print("User features is not existed")
        try:
            self.item_category_features = np.zeros((int(self.items_profiles["item"].max() + 1), max_n_categories))
            self.item_numeric_features = np.zeros((int(self.items_profiles["item"].max() + 1), len(self.item_numeric_columns)))
            self.item_category_features[self.items_profiles["item"].values.astype(int)] = items_profiles_category
            self.item_numeric_features[self.items_profiles["item"].values.astype(int)] = self.items_profiles[self.item_numeric_columns]
        except:
            print("Item features is not existed")
    def load_full_cf(self, task_id):
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file, task_id)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file, task_id)
        self.cf_test_cold_data, self.test_cold_user_dict = self.load_cf(self.test_cold_file, task_id)
        if task_id != self.pretrain_task_id:
            _, self.train_data_masking_test_user_dict = self.load_cf(self.train_data_masking_test_file, task_id)
        else:
            self.train_data_masking_test_user_dict = {0 : [0]}
        item_to_task_id_group_by_items = item_to_task_id.groupby('new_id', as_index = False).agg(lambda x : '|'.join([str(elem) for elem in x]))
        item_to_task_id_group_by_items['check_out_task'] = item_to_task_id_group_by_items['task_id'].apply(lambda x : (str(task_id) not in x))
        self.items_out_task = item_to_task_id_group_by_items["new_id"][item_to_task_id_group_by_items["check_out_task"]].values
    def load_cf(self, filename, task_id):
        lines = pd.read_csv(filename, sep = "|", names = ["user", "items"])
        if self.n_interactions > 0:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")][::-1][:self.n_interactions])
        else:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
        lines["user"] = lines["user"].apply(lambda x : int(x))
        self.user_dict_raw = lines.copy()
        true_task_item = self.item_to_task_id[self.item_to_task_id["task_id"] == task_id]
        true_task_item = dict(true_task_item.values)
        #print(lines)
        lines["items"] = lines["items"].apply(lambda x : [elem for elem in x if elem in true_task_item])
        lines["n_interactions"] = lines["items"].apply(lambda x : len(x))
        lines = lines[lines["n_interactions"] > 0]
        lines = lines.drop(["n_interactions"], axis = 1)
        user_dict = dict(lines.values)
        #print(lines)
        if self.full_sampling:
            try:
                self.n_interactions_max = max(int(lines["items"].apply(lambda x : len(x)).max()), self.n_interactions_max)
            except:
                return (None, None), {}
        lines["items"] =  lines["items"].apply(lambda x : " ".join([str(elem) for elem in x]))
        lines["user"] = lines["user"].apply(lambda x : str(x))
        lines["user"] = lines["user"] + " " + lines["items"].apply(lambda x : str(len(x.split(" "))))
        lines["user"] = lines["user"].apply(lambda x : " ".join(list(np.full((int(x.split(" ")[1]),), x.split(" ")[0]))))
        user = np.array(" ".join(lines["user"].values).split(" "), dtype=np.int32)
        #print(user)
        item = np.array(" ".join(lines["items"].values).split(" "), dtype=np.int32)
        return (user, item), user_dict
    def load_cf_basic_full(self):
        self.cf_train_data, self.train_user_dict = self.load_cf_basic(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf_basic(self.test_file)
        self.cf_test_cold_data, self.test_cold_user_dict = self.load_cf_basic(self.test_cold_file)
        _, self.train_data_masking_test_user_dict = self.load_cf_basic(self.train_data_masking_test_file)
    def load_cf_basic(self, filename):
        lines = pd.read_csv(filename, sep = "|", names = ["user", "items"])
        if self.n_interactions > 0:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")][::-1][:self.n_interactions])
        else:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
        lines["user"] = lines["user"].apply(lambda x : int(x))
        self.user_dict_raw = lines.copy()
        user_dict = dict(lines.values)
        #print(lines)
        if self.full_sampling:
            try:
                self.n_interactions_max = max(int(lines["items"].apply(lambda x : len(x)).max()), self.n_interactions_max)
            except:
                return (None, None), {}
        lines["items"] =  lines["items"].apply(lambda x : " ".join([str(elem) for elem in x]))
        lines["user"] = lines["user"].apply(lambda x : str(x))
        lines["user"] = lines["user"] + " " + lines["items"].apply(lambda x : str(len(x.split(" "))))
        lines["user"] = lines["user"].apply(lambda x : " ".join(list(np.full((int(x.split(" ")[1]),), x.split(" ")[0]))))
        user = np.array(" ".join(lines["user"].values).split(" "), dtype=np.int32)
        print(user)
        item = np.array(" ".join(lines["items"].values).split(" "), dtype=np.int32)
        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0]))
        self.n_users = max(self.n_users, max(self.cf_test_cold_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1]))
        self.n_items = max(self.n_items, max(self.cf_test_cold_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        self.max_id = [max(self.cf_train_data[0]), max(self.cf_train_data[1])]


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep = ' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data
        
    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)
        padding_zeros = []
        if n_sample_pos_items > n_pos_items:
            padding_zeros = np.zeros((n_sample_pos_items - n_pos_items,)) - 1
            padding_zeros = list(padding_zeros)
            n_sample_pos_items = n_pos_items
        random_item = np.random.permutation(n_pos_items)[:n_sample_pos_items]
        sample_pos_items = np.asarray(pos_items)[random_item] 
        return padding_zeros + list(sample_pos_items)
    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]
        sample_neg_items = []
        padding_zeros = []
        if n_sample_neg_items > len(pos_items):
            padding_zeros = np.zeros((n_sample_neg_items - len(pos_items),)) - 1
            padding_zeros = list(padding_zeros)
            n_sample_neg_items = len(pos_items)
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if self.full_sampling:
                if neg_item_id not in pos_items:
                    sample_neg_items.append(neg_item_id)
                continue
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return padding_zeros + sample_neg_items
    def sample_user(self, user_dict, user_id, n_sample):
        pos_items = user_dict[user_id]
        padding_zeros = []
        if n_sample > len(pos_items):
            padding_zeros = np.zeros((n_sample - len(pos_items),)) - 1
            padding_zeros = list(padding_zeros)
            n_sample = len(pos_items)
        return padding_zeros + list(np.full((n_sample,), user_id))
    def generate_cf_batch(self, user_dict, batch_size, reranking_loss = False):
        exist_users = user_dict.keys()
        #print(len(exist_users))
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        #n_pos = 5
        n_pos = 1
        if self.full_sampling:
            n_pos = self.n_interactions_max
        if reranking_loss:
            n_pos = len(user_dict.values()[0])
        n_neg = n_pos
        n_users = n_pos
        batch_pos_item = [self.sample_pos_items_for_u(user_dict, u, n_pos) for u in batch_user]
        batch_pos_item = np.asarray(batch_pos_item).reshape(-1)
        batch_pos_item = batch_pos_item[batch_pos_item >= 0]
        batch_pos_item = torch.LongTensor(batch_pos_item)
        
        batch_user = [self.sample_user(user_dict, u, n_users) for u in batch_user]
        batch_user = np.asarray(batch_user)
        batch_user = batch_user[batch_user >= 0]
        batch_user = torch.LongTensor(batch_user)
        
        if reranking_loss:
            return batch_user, batch_pos_item
        
        batch_neg_item = [self.sample_neg_items_for_u(user_dict, u, n_neg) for u in batch_user]
        batch_neg_item = np.asarray(batch_neg_item).reshape(-1)
        batch_neg_item = batch_neg_item[batch_neg_item >= 0]
        batch_neg_item = torch.LongTensor(batch_neg_item)
        
        return batch_user, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
        batch_relation_tail = [self.sample_pos_triples_for_h(kg_dict, h, 1)[0] +
                               self.sample_pos_triples_for_h(kg_dict, h, 1)[1] +
                               self.sample_neg_triples_for_h(kg_dict, h, self.sample_pos_triples_for_h(kg_dict, h, 1)[0], 1, highest_neg_idx)
                               for h in batch_head]
        batch_relation_tail = np.asarray(batch_relation_tail)
        batch_relation = batch_relation_tail[:, 0]
        batch_pos_tail = batch_relation_tail[:, 1]
        batch_neg_tail = batch_relation_tail[:, 2]
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embed_dim
        assert self.item_pre_embed.shape[1] == self.args.embed_dim

        
        
        
class DataLoaderBase_MultiTask_full(object):

    def __init__(self, args, logging):
        self.n_interactions_max = 1
        self.pretrain_task_id = args.pretrain_task_id
        self.full_sampling = args.full_sampling
        self.args = args
        self.n_interactions = args.n_interactions
        self.data_centric = args.data_centric
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.test_cold_file = os.path.join(self.data_dir, 'test_cold.txt')
        self.item_to_task_id = pd.read_csv(os.path.join(self.data_dir, "item_to_task_id.csv"))
        self.item_to_task_id["new_id"] = self.item_to_task_id["new_id"].astype(int)
        self.item_to_task_id["task_id"] = self.item_to_task_id["task_id"].astype(int)
        self.train_data_masking_test_file = os.path.join(self.data_dir, 'train_for_test.txt')
        try:
            self.user_features = pd.read_csv(os.path.join(self.data_dir, "user_features.csv"))
        except:
            self.user_features = None
        try:
            self.item_features = pd.read_csv(os.path.join(self.data_dir, "item_features.csv"))
        except:
            self.item_features = None   
        print(self.item_features)
        self.load_cf_basic_full()
        self.statistic_cf()
        task_id = [int(elem) for elem in args.task_ids.split(',')][0]
        self.load_full_cf(task_id)
        print("Max num interactions is {}".format(self.n_interactions_max))
        self.user_category_columns = ["gender", "province_name"]
        #self.user_numeric_columns = ["age", "n_profiles"]
        self.user_numeric_columns = []
        self.item_category_columns = ["all_categories"]
        #self.item_numeric_columns = ["duration", "imdb_rating", "release_year"]
        self.item_numeric_columns = []
        self.user_n_interactions = None
        if self.data_centric == 1.:
            scale = 0.5
            self.user_n_interactions = self.user_features["user"].apply(lambda x : (len(self.train_user_dict[int(x)]) * scale if int(x) in self.train_user_dict else 0)).values
        def category_to_id(data, column):
            to_new_id = dict(np.hstack((data[column][pd.isnull(data[column]) == False].unique().reshape(-1, 1),
                                        np.linspace(2, data[column].nunique() + 1, data[column].nunique()).astype(int).reshape(-1, 1))))
            data[column][pd.isnull(data[column]) == False] = data[column][pd.isnull(data[column]) == False].apply(lambda x : to_new_id[x])
            data[column][pd.isnull(data[column])] = 1
            return data, to_new_id
        all_categorys_to_id = {}
        for column in self.user_category_columns:
            if self.user_features is None:
                break
            self.user_features, new_category_to_id = category_to_id(self.user_features, column)
            all_categorys_to_id[column] = new_category_to_id
        self.max_size_embedding_category = [0]
        self.max_len_category = [len(self.user_category_columns)]
        self.n_category = [len(self.user_category_columns)]
        self.n_numeric = [len(self.user_numeric_columns), len(self.item_numeric_columns)]
        def preprocess_profiles_user(recent_profiles, category_columns, numeric_columns, max_size_embedding_category, start_index = 3, max_len_category = None):
            new_max_size_category = max_size_embedding_category
            profiles = recent_profiles.copy()
            try:
                scaler = StandardScaler() 
                profiles[numeric_columns] = scaler.fit_transform(profiles[numeric_columns])
            except:
                print("Numeric features are not existed")
            columns_to_numeric_id = dict([(column, {}) for column in category_columns])
            for i, column in enumerate(category_columns):
                n_distinct_val = profiles[column].nunique()
                columns_to_numeric_id[column] = dict(np.hstack((profiles[column].unique().reshape(-1, 1), 
                                                                (np.linspace(0, n_distinct_val - 1, n_distinct_val) * max_len_category + i + start_index).reshape(-1, 1).astype(int))))
                new_max_size_category = max(new_max_size_category, int((n_distinct_val - 1) * max_len_category + i + start_index))
                columns_to_numeric_id[column][0] = 0
                profiles[column] = profiles[column].apply(lambda x : columns_to_numeric_id[column][x])
            return profiles, new_max_size_category
        def preprocess_profiles_item(recent_profiles, category_columns, numeric_columns):
            profiles = recent_profiles.copy()
            try:
                scaler = StandardScaler() 
                profiles[numeric_columns] = scaler.fit_transform(profiles[numeric_columns])
            except:
                print("Numeric features are not existed")
            columns_to_numeric_id = dict([(column, {}) for column in category_columns])
            profiles_categories = profiles[category_columns[0]].apply(lambda x : [int(elem) for elem in str(x).split("\t")])
            max_n_categories = int(profiles_categories.apply(lambda x : len(x)).max())
            profiles_categories = pad_sequences(profiles_categories.values, max_n_categories)
            new_max_size_category = np.max(profiles_categories.reshape(-1))
            return profiles, new_max_size_category, profiles_categories, max_n_categories 
        if self.use_pretrain == 1:
            self.load_pretrained_data()
        if self.user_features is not None:
            self.users_profiles, self.max_size_embedding_category[0] =  preprocess_profiles_user(self.user_features, self.user_category_columns, self.user_numeric_columns, self.max_size_embedding_category[0], 3, self.max_len_category[0])
        if self.item_features is not None:
            self.items_profiles, max_size_items_embedding_category, items_profiles_category, max_n_categories =  preprocess_profiles_item(self.item_features, self.item_category_columns, self.item_numeric_columns)
        if self.user_features is not None or self.item_features is not None:
            self.max_len_category.append(max_n_categories)
            self.n_category.append(max_n_categories)
            self.max_size_embedding_category.append(max_size_items_embedding_category)
        self.user_category_features = None
        self.user_numeric_features = None
        self.item_category_features = None
        self.item_numeric_features = None
        try:
            self.user_category_features = np.zeros((int(self.users_profiles["user"].max() + 1), len(self.user_category_columns)))
            self.user_numeric_features = np.zeros((int(self.users_profiles["user"].max() + 1), len(self.user_numeric_columns)))
            self.user_category_features[self.users_profiles["user"].values.astype(int)] = self.users_profiles[self.user_category_columns]
            self.user_numeric_features[self.users_profiles["user"].values.astype(int)] = self.users_profiles[self.user_numeric_columns]
        except:
            print("User features is not existed")
        try:
            self.item_category_features = np.zeros((int(self.items_profiles["item"].max() + 1), max_n_categories))
            self.item_numeric_features = np.zeros((int(self.items_profiles["item"].max() + 1), len(self.item_numeric_columns)))
            self.item_category_features[self.items_profiles["item"].values.astype(int)] = items_profiles_category
            self.item_numeric_features[self.items_profiles["item"].values.astype(int)] = self.items_profiles[self.item_numeric_columns]
        except:
            print("Item features is not existed")
    def load_full_cf(self, task_id):
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file, task_id)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file, task_id)
        self.cf_test_cold_data, self.test_cold_user_dict = self.load_cf(self.test_cold_file, task_id)
        if task_id != self.pretrain_task_id:
            _, self.train_data_masking_test_user_dict = self.load_cf(self.train_data_masking_test_file, task_id)
        else:
            self.train_data_masking_test_user_dict = {0 : [0]}
        self.items_out_task = self.item_to_task_id["new_id"][self.item_to_task_id["task_id"] == int(task_id + 1) % 2].values
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        self.max_id = [max(self.cf_train_data[0]), max(self.cf_train_data[1])]

    def load_cf(self, filename, task_id):
        lines = pd.read_csv(filename, sep = "|", names = ["user", "items"])
        if self.n_interactions > 0:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")][::-1][:self.n_interactions])
        else:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
        lines["user"] = lines["user"].apply(lambda x : int(x))
        self.user_dict_raw = lines.copy()
        true_task_item = self.item_to_task_id[self.item_to_task_id["task_id"] == task_id]
        true_task_item = dict(true_task_item.values)
        #print(lines)
        lines["items"] = lines["items"].apply(lambda x : [elem for elem in x if elem in true_task_item])
        lines["n_interactions"] = lines["items"].apply(lambda x : len(x))
        lines = lines[lines["n_interactions"] > 0]
        lines = lines.drop(["n_interactions"], axis = 1)
        user_dict = dict(lines.values)
        #print(lines)
        if self.full_sampling:
            try:
                self.n_interactions_max = max(int(lines["items"].apply(lambda x : len(x)).max()), self.n_interactions_max)
            except:
                return (None, None), {}
        lines["items"] =  lines["items"].apply(lambda x : " ".join([str(elem) for elem in x]))
        lines["user"] = lines["user"].apply(lambda x : str(x))
        lines["user"] = lines["user"] + " " + lines["items"].apply(lambda x : str(len(x.split(" "))))
        lines["user"] = lines["user"].apply(lambda x : " ".join(list(np.full((int(x.split(" ")[1]),), x.split(" ")[0]))))
        user = np.array(" ".join(lines["user"].values).split(" "), dtype=np.int32)
        #print(user)
        item = np.array(" ".join(lines["items"].values).split(" "), dtype=np.int32)
        return (user, item), user_dict
    def load_cf_basic_full(self, n_entities):
        self.cf_train_data, self.train_user_dict = self.load_cf_basic(self.train_file, n_entities)
        self.cf_test_data, self.test_user_dict = self.load_cf_basic(self.test_file, n_entities)
        self.cf_test_cold_data, self.test_cold_user_dict = self.load_cf_basic(self.test_cold_file, n_entities)
        _, self.train_data_masking_test_user_dict = self.load_cf_basic(self.train_data_masking_test_file, n_entities)
    def load_cf_basic(self, filename, n_entities):
        lines = pd.read_csv(filename, sep = "|", names = ["user", "items"])
        if self.n_interactions > 0:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")][::-1][:self.n_interactions])
        else:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
        lines["user"] = lines["user"].apply(lambda x : int(x))
        self.user_dict_raw = lines.copy()
        user_dict = dict(lines.values)
        #print(lines)
        if self.full_sampling:
            try:
                self.n_interactions_max = max(int(lines["items"].apply(lambda x : len(x)).max()), self.n_interactions_max)
            except:
                return (None, None), {}
        lines["items"] =  lines["items"].apply(lambda x : " ".join([str(elem) for elem in x]))
        lines["user"] = lines["user"].apply(lambda x : str(x))
        lines["user"] = lines["user"] + " " + lines["items"].apply(lambda x : str(len(x.split(" "))))
        lines["user"] = lines["user"].apply(lambda x : " ".join(list(np.full((int(x.split(" ")[1]),), x.split(" ")[0]))))
        user = np.array(" ".join(lines["user"].values).split(" "), dtype=np.int32)
        print(user)
        item = np.array(" ".join(lines["items"].values).split(" "), dtype=np.int32)
        user_dict = {k + n_entities: np.unique(v).astype(np.int32) for k, v in user_dict.items()}
        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0]))
        self.n_users = max(self.n_users, max(self.cf_test_cold_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1]))
        self.n_items = max(self.n_items, max(self.cf_test_cold_data[1])) + 1

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep = ' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data
        
    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)
        padding_zeros = []
        if n_sample_pos_items > n_pos_items:
            padding_zeros = np.zeros((n_sample_pos_items - n_pos_items,)) - 1
            padding_zeros = list(padding_zeros)
            n_sample_pos_items = n_pos_items
        random_item = np.random.permutation(n_pos_items)[:n_sample_pos_items]
        sample_pos_items = np.asarray(pos_items)[random_item] 
        return padding_zeros + list(sample_pos_items)
    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]
        sample_neg_items = []
        padding_zeros = []
        if n_sample_neg_items > len(pos_items):
            padding_zeros = np.zeros((n_sample_neg_items - len(pos_items),)) - 1
            padding_zeros = list(padding_zeros)
            n_sample_neg_items = len(pos_items)
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if self.full_sampling:
                if neg_item_id not in pos_items:
                    sample_neg_items.append(neg_item_id)
                continue
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return padding_zeros + sample_neg_items
    def sample_user(self, user_dict, user_id, n_sample):
        pos_items = user_dict[user_id]
        padding_zeros = []
        if n_sample > len(pos_items):
            padding_zeros = np.zeros((n_sample - len(pos_items),)) - 1
            padding_zeros = list(padding_zeros)
            n_sample = len(pos_items)
        return padding_zeros + list(np.full((n_sample,), user_id))
    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = user_dict.keys()
        #print(len(exist_users))
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        #n_pos = 5
        n_pos = 1
        if self.full_sampling:
            n_pos = self.n_interactions_max
        n_neg = n_pos
        n_users = n_pos
        batch_pos_item = [self.sample_pos_items_for_u(user_dict, u, n_pos) for u in batch_user]
        batch_pos_item = np.asarray(batch_pos_item).reshape(-1)
        batch_pos_item = batch_pos_item[batch_pos_item >= 0]
        batch_neg_item = [self.sample_neg_items_for_u(user_dict, u, n_neg) for u in batch_user]
        batch_neg_item = np.asarray(batch_neg_item).reshape(-1)
        batch_neg_item = batch_neg_item[batch_neg_item >= 0]
        batch_user = [self.sample_user(user_dict, u, n_users) for u in batch_user]
        batch_user = np.asarray(batch_user)
        batch_user = batch_user[batch_user >= 0]
        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
        batch_relation_tail = [self.sample_pos_triples_for_h(kg_dict, h, 1)[0] +
                               self.sample_pos_triples_for_h(kg_dict, h, 1)[1] +
                               self.sample_neg_triples_for_h(kg_dict, h, self.sample_pos_triples_for_h(kg_dict, h, 1)[0], 1, highest_neg_idx)
                               for h in batch_head]
        batch_relation_tail = np.asarray(batch_relation_tail)
        batch_relation = batch_relation_tail[:, 0]
        batch_pos_tail = batch_relation_tail[:, 1]
        batch_neg_tail = batch_relation_tail[:, 2]
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embed_dim
        assert self.item_pre_embed.shape[1] == self.args.embed_dim
        
        
        
class DataLoaderBase_MultiTask_full_2(object):

    def __init__(self, args, logging):
        self.check_multi_task = args.check_multi_task
        self.n_interactions_max = 1
        self.pretrain_task_id = args.pretrain_task_id
        self.full_sampling = args.full_sampling
        self.args = args
        self.n_interactions = args.n_interactions
        self.data_centric = args.data_centric
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.val_file = os.path.join(self.data_dir, 'val.txt')
        self.test_cold_file = os.path.join(self.data_dir, 'test_cold.txt')
        self.item_to_task_id = pd.read_csv(os.path.join(self.data_dir, "item_to_task_id.csv"))
        self.item_to_task_id["new_id"] = self.item_to_task_id["new_id"].astype(int)
        self.item_to_task_id["task_id"] = self.item_to_task_id["task_id"].astype(int)
        self.train_data_masking_test_file = os.path.join(self.data_dir, 'train_for_test.txt')
        self.check_initalize_n_max_clusters = False
        try:
            self.user_features = pd.read_csv(os.path.join(self.data_dir, "user_features.csv"))
        except:
            self.user_features = None
        try:
            self.item_features = pd.read_csv(os.path.join(self.data_dir, "item_features.csv"))
        except:
            self.item_features = None  
        try:
            self.users_group = pd.read_csv(os.path.join(self.data_dir, "user_to_their_group.csv"))
            self.users_group = dict(self.users_group.values)
            self.n_clusters_users = max(list(self.users_group.values())) + 1
        except:
            self.users_group = None
            self.n_clusters_users = None
        try:
            self.items_group = pd.read_csv(os.path.join(self.data_dir, "item_to_their_group.csv"))
            self.items_group = dict(self.items_group.values)
            self.n_clusters_items = max(list(self.items_group.values())) + 1
        except:
            self.items_group = None
            self.n_clusters_items = None
        print(self.item_features)
        self.load_cf_basic_full()
        self.statistic_cf()
        task_id = [int(elem) for elem in args.task_ids.split(',')][0]
        self.load_full_cf(task_id, 0)
        print("Max num interactions is {}".format(self.n_interactions_max))
        #self.user_category_columns = ["gender", "province_name"]
        #self.user_category_columns = ["gender"]
        self.user_category_columns = []
        #self.user_numeric_columns = ["age", "n_profiles"]
        self.user_numeric_columns = []
        #self.item_category_columns = ["all_categories"]
        self.item_category_columns = []
        #self.item_numeric_columns = ["duration", "imdb_rating", "release_year"]
        self.item_numeric_columns = []
        self.user_n_interactions = None
        if self.data_centric == 1.:
            scale = 0.5
            self.user_n_interactions = self.user_features["user"].apply(lambda x : (len(self.train_user_dict[int(x)]) * scale if int(x) in self.train_user_dict else 0)).values
        def category_to_id(data, column):
            to_new_id = dict(np.hstack((data[column][pd.isnull(data[column]) == False].unique().reshape(-1, 1),
                                        np.linspace(2, data[column].nunique() + 1, data[column].nunique()).astype(int).reshape(-1, 1))))
            data[column][pd.isnull(data[column]) == False] = data[column][pd.isnull(data[column]) == False].apply(lambda x : to_new_id[x])
            data[column][pd.isnull(data[column])] = 1
            return data, to_new_id
        all_categorys_to_id = {}
        for column in self.user_category_columns:
            if self.user_features is None:
                break
            self.user_features, new_category_to_id = category_to_id(self.user_features, column)
            all_categorys_to_id[column] = new_category_to_id
        self.max_size_embedding_category = [0]
        self.max_len_category = [len(self.user_category_columns)]
        self.n_category = [len(self.user_category_columns)]
        self.n_numeric = [len(self.user_numeric_columns), len(self.item_numeric_columns)]
        def preprocess_profiles_user(recent_profiles, category_columns, numeric_columns, max_size_embedding_category, start_index = 3, max_len_category = None):
            new_max_size_category = max_size_embedding_category
            profiles = recent_profiles.copy()
            try:
                scaler = StandardScaler() 
                profiles[numeric_columns] = scaler.fit_transform(profiles[numeric_columns])
            except:
                print("Numeric features are not existed")
            columns_to_numeric_id = dict([(column, {}) for column in category_columns])
            for i, column in enumerate(category_columns):
                n_distinct_val = profiles[column].nunique()
                columns_to_numeric_id[column] = dict(np.hstack((profiles[column].unique().reshape(-1, 1), 
                                                                (np.linspace(0, n_distinct_val - 1, n_distinct_val) * max_len_category + i + start_index).reshape(-1, 1).astype(int))))
                new_max_size_category = max(new_max_size_category, int((n_distinct_val - 1) * max_len_category + i + start_index))
                columns_to_numeric_id[column][0] = 0
                profiles[column] = profiles[column].apply(lambda x : columns_to_numeric_id[column][x])
            return profiles, new_max_size_category
        def preprocess_profiles_item(recent_profiles, category_columns, numeric_columns):
            profiles = recent_profiles.copy()
            try:
                scaler = StandardScaler() 
                profiles[numeric_columns] = scaler.fit_transform(profiles[numeric_columns])
            except:
                print("Numeric features are not existed")
            columns_to_numeric_id = dict([(column, {}) for column in category_columns])
            profiles_categories = profiles[category_columns[0]].apply(lambda x : [int(elem) for elem in str(x).split("\t")])
            max_n_categories = int(profiles_categories.apply(lambda x : len(x)).max())
            profiles_categories = pad_sequences(profiles_categories.values, max_n_categories)
            new_max_size_category = np.max(profiles_categories.reshape(-1))
            return profiles, new_max_size_category, profiles_categories, max_n_categories 
        if self.use_pretrain == 1:
            self.load_pretrained_data()
        if self.user_features is not None:
            self.users_profiles, self.max_size_embedding_category[0] =  preprocess_profiles_user(self.user_features, self.user_category_columns, self.user_numeric_columns, self.max_size_embedding_category[0], 3, self.max_len_category[0])
        if self.item_features is not None:
            self.items_profiles, max_size_items_embedding_category, items_profiles_category, max_n_categories =  preprocess_profiles_item(self.item_features, self.item_category_columns, self.item_numeric_columns)
        if self.user_features is not None or self.item_features is not None:
            try:
              self.max_len_category.append(max_n_categories)
              self.n_category.append(max_n_categories)
              self.max_size_embedding_category.append(max_size_items_embedding_category)
            except:
              self.max_len_category.append(0)
              self.n_category.append(0)
              self.max_size_embedding_category.append(0)
        self.user_category_features = None
        self.user_numeric_features = None
        self.item_category_features = None
        self.item_numeric_features = None
        try:
            self.user_category_features = np.zeros((int(self.users_profiles["user"].max() + 1), len(self.user_category_columns)))
            self.user_numeric_features = np.zeros((int(self.users_profiles["user"].max() + 1), len(self.user_numeric_columns)))
            self.user_category_features[self.users_profiles["user"].values.astype(int)] = self.users_profiles[self.user_category_columns]
            self.user_numeric_features[self.users_profiles["user"].values.astype(int)] = self.users_profiles[self.user_numeric_columns]
            self.n_users = max(self.n_users, int(self.users_profiles["user"].max() + 1))
            print('cc')
        except:
            print("User features is not existed")
        try:
            self.item_category_features = np.zeros((int(self.items_profiles["item"].max() + 1), max_n_categories))
            self.item_numeric_features = np.zeros((int(self.items_profiles["item"].max() + 1), len(self.item_numeric_columns)))
            self.item_category_features[self.items_profiles["item"].values.astype(int)] = items_profiles_category
            self.item_numeric_features[self.items_profiles["item"].values.astype(int)] = self.items_profiles[self.item_numeric_columns]
            self.n_items = max(self.n_items, int(self.items_profiles["item"].max() + 1))
            print('cc')
        except:
            print("Item features is not existed")
        self.n_max_clusters_users_items = []
    def load_full_cf(self, task_id, n_entities):
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file, task_id, n_entities)
        self.cf_val_data, self.val_user_dict = self.load_cf(self.val_file, task_id, n_entities)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file, task_id, n_entities)
        self.cf_test_cold_data, self.test_cold_user_dict = self.load_cf(self.test_cold_file, task_id, n_entities)
        if task_id != self.pretrain_task_id:
            _, self.train_data_masking_test_user_dict = self.load_cf(self.train_data_masking_test_file, task_id, n_entities)
        else:
            self.train_data_masking_test_user_dict = {0 : [0]}
        item_to_task_id_group_by_items = self.item_to_task_id.groupby('new_id', as_index = False).agg(lambda x : '|'.join([str(elem) for elem in x]))
        item_to_task_id_group_by_items['check_out_task'] = item_to_task_id_group_by_items['task_id'].apply(lambda x : (str(task_id) not in x))
        self.items_out_task = item_to_task_id_group_by_items["new_id"][item_to_task_id_group_by_items["check_out_task"]].values
        self.items_in_task = self.item_to_task_id["new_id"][self.item_to_task_id["task_id"] == task_id].values
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        self.max_id = [max(self.cf_train_data[0]), max(self.cf_train_data[1])]
    def load_cf(self, filename, task_id, n_entities):
        lines = pd.read_csv(filename, sep = "|", names = ["user", "items"])
        if self.n_interactions > 0:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")][::-1][:self.n_interactions])
        else:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
        lines["user"] = lines["user"].apply(lambda x : int(x))
        self.user_dict_raw = lines.copy()
        if self.check_multi_task:
            true_task_item = self.item_to_task_id[self.item_to_task_id["task_id"] == task_id]
            true_task_item = dict(true_task_item.values)
            #print(lines)
            lines["items"] = lines["items"].apply(lambda x : [elem for elem in x if elem in true_task_item])
            lines["n_interactions"] = lines["items"].apply(lambda x : len(x))
            lines = lines[lines["n_interactions"] > 0]
            lines = lines.drop(["n_interactions"], axis = 1)
        user_dict = dict(lines.values)
        #print(lines)
        if self.full_sampling:
            try:
                self.n_interactions_max = max(int(lines["items"].apply(lambda x : len(x)).max()), self.n_interactions_max)
            except:
                return (None, None), {}
        try:
          lines["items"] =  lines["items"].apply(lambda x : " ".join([str(elem) for elem in x]))
          lines["user"] = lines["user"].apply(lambda x : str(x))
          lines["user"] = lines["user"] + " " + lines["items"].apply(lambda x : str(len(x.split(" "))))
          lines["user"] = lines["user"].apply(lambda x : " ".join(list(np.full((int(x.split(" ")[1]),), x.split(" ")[0]))))
          user = np.array(" ".join(lines["user"].values).split(" "), dtype=np.int32)
          #print(user)
          item = np.array(" ".join(lines["items"].values).split(" "), dtype=np.int32)
          user_dict = {k + n_entities: np.unique(v).astype(np.int32) for k, v in user_dict.items()}
          return (user, item), user_dict
        except:
          return (None, None), {}
    def load_cf_basic_full(self, n_entities = None):
        self.cf_train_data, self.train_user_dict = self.load_cf_basic(self.train_file, n_entities)
        self.cf_val_data, self.val_user_dict = self.load_cf_basic(self.val_file, n_entities)
        self.cf_test_data, self.test_user_dict = self.load_cf_basic(self.test_file, n_entities)
        self.cf_test_cold_data, self.test_cold_user_dict = self.load_cf_basic(self.test_cold_file, n_entities)
        _, self.train_data_masking_test_user_dict = self.load_cf_basic(self.train_data_masking_test_file, n_entities)
    def load_cf_basic(self, filename, n_entities = None):
        lines = pd.read_csv(filename, sep = "|", names = ["user", "items"])
        if not self.check_initalize_n_max_clusters:
            new_lines = lines.copy()
            new_lines["items"] =  new_lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
            new_lines = new_lines.explode('items')
            new_lines['item'] = new_lines['items'].copy()
            new_lines = new_lines.drop(['items'], axis = 1)
            task_ids = list(set(self.item_to_task_id['task_id'].values))
            sum_n_users_if_specify_each_task = 0
            sum_n_items_if_specify_each_task = 0
            n_true_users = new_lines['user'].nunique()
            n_true_items = new_lines['item'].nunique()
            #print('con cak')
            #print(n_true_users)
            #print(n_true_items)
            #print(task_ids)
            for i in task_ids:
                get_items_in_task_i = self.item_to_task_id['new_id'][self.item_to_task_id['task_id'] == i].values
                new_lines_in_task_i = new_lines[new_lines['item'].isin(get_items_in_task_i)]
                sum_n_users_if_specify_each_task += new_lines_in_task_i['user'].nunique()
                sum_n_items_if_specify_each_task += new_lines_in_task_i['item'].nunique()
            n_maximum_clusters_for_args = int((sum_n_users_if_specify_each_task + sum_n_items_if_specify_each_task - n_true_users - n_true_items)/(2 * len(task_ids)))
            print('The maximum of the number of clusters for all entities is {}'.format(n_maximum_clusters_for_args))
            if n_maximum_clusters_for_args <= self.args.n_task_masks:
                self.args.n_task_masks = n_maximum_clusters_for_args - 1
                if self.args.n_task_masks < 0:
                    print('Warning, the number overlapped users and items are less than two')
                    self.args.n_task_masks += 1
            self.check_initalize_n_max_clusters = True
                
        if self.n_interactions > 0:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")][::-1][:self.n_interactions])
        else:
            lines["items"] =  lines["items"].apply(lambda x : [int(elem) for elem in str(x).split(" ")])
        lines["user"] = lines["user"].apply(lambda x : int(x))
        self.user_dict_raw = lines.copy()
        user_dict = dict(lines.values)
        #print(lines)
        if self.full_sampling:
            try:
                self.n_interactions_max = max(int(lines["items"].apply(lambda x : len(x)).max()), self.n_interactions_max)
            except:
                return (None, None), {}
        try:
          lines["items"] =  lines["items"].apply(lambda x : " ".join([str(elem) for elem in x]))
          lines["user"] = lines["user"].apply(lambda x : str(x))
          lines["user"] = lines["user"] + " " + lines["items"].apply(lambda x : str(len(x.split(" "))))
          lines["user"] = lines["user"].apply(lambda x : " ".join(list(np.full((int(x.split(" ")[1]),), x.split(" ")[0]))))
          user = np.array(" ".join(lines["user"].values).split(" "), dtype=np.int32)
          print(user)
          item = np.array(" ".join(lines["items"].values).split(" "), dtype=np.int32)
          if n_entities is not None:
                user_dict = {k + n_entities: np.unique(v).astype(np.int32) for k, v in user_dict.items()}
          return (user, item), user_dict
        except:
          return (None, None), {}


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0]))
        try:
          self.n_users = max(self.n_users, max(self.cf_test_cold_data[0]))
        except:
          print('No test cold')
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1]))
        try:
          self.n_items = max(self.n_items, max(self.cf_test_cold_data[1]))
        except:
          print('No test codld')
        try:
          self.n_users = max(self.n_users, max(self.cf_val_data[0]))
        except:
          print('No val dataset')
        try:
          self.n_items = max(self.n_items, max(self.cf_val_data[1]))
        except:
          print('No val dataset')
        self.n_users += 1
        self.n_items += 1

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep = ' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data
        
    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        if self.args.using_val_data_for_evaluating_mask_performance:
            pos_items = user_dict[user_id][:-1]
        else:
            pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)
        padding_zeros = []
        if n_sample_pos_items > n_pos_items:
            padding_zeros = np.zeros((n_sample_pos_items - n_pos_items,)) - 1
            padding_zeros = list(padding_zeros)
            n_sample_pos_items = n_pos_items
        random_item = np.random.permutation(n_pos_items)[:n_sample_pos_items]
        sample_pos_items = np.asarray(pos_items)[random_item] 
        return padding_zeros + list(sample_pos_items)
    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        if self.args.using_val_data_for_evaluating_mask_performance:
            pos_items = user_dict[user_id][:-1]
        else:
            pos_items = user_dict[user_id]
        sample_neg_items = []
        padding_zeros = []
        if n_sample_neg_items > len(pos_items):
            padding_zeros = np.zeros((n_sample_neg_items - len(pos_items),)) - 1
            padding_zeros = list(padding_zeros)
            n_sample_neg_items = len(pos_items)
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if self.full_sampling:
                if neg_item_id not in pos_items and neg_item_id in self.items_in_task:
                    sample_neg_items.append(neg_item_id)
                continue
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items and neg_item_id in self.items_in_task:
                sample_neg_items.append(neg_item_id)
        return padding_zeros + sample_neg_items
    def sample_user(self, user_dict, user_id, n_sample):
        if self.args.using_val_data_for_evaluating_mask_performance:
            pos_items = user_dict[user_id][:-1]
        else:
            pos_items = user_dict[user_id]
        padding_zeros = []
        if n_sample > len(pos_items):
            padding_zeros = np.zeros((n_sample - len(pos_items),)) - 1
            padding_zeros = list(padding_zeros)
            n_sample = len(pos_items)
        return padding_zeros + list(np.full((n_sample,), user_id))
    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = list(user_dict.keys())
        #print(len(exist_users))
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        #n_pos = 5
        #n_pos = 1
        n_pos = 10
        if self.full_sampling:
            n_pos = self.n_interactions_max
        #n_pos = 10
        n_neg = n_pos
        n_users = n_pos
        batch_pos_item = [self.sample_pos_items_for_u(user_dict, u, n_pos) for u in batch_user]
        batch_pos_item = np.asarray(batch_pos_item).reshape(-1)
        batch_pos_item = batch_pos_item[batch_pos_item >= 0]
        batch_neg_item = [self.sample_neg_items_for_u(user_dict, u, n_neg) for u in batch_user]
        batch_neg_item = np.asarray(batch_neg_item).reshape(-1)
        batch_neg_item = batch_neg_item[batch_neg_item >= 0]
        batch_user = [self.sample_user(user_dict, u, n_users) for u in batch_user]
        batch_user = np.asarray(batch_user)
        batch_user = batch_user[batch_user >= 0]
        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
        batch_relation_tail = [self.sample_pos_triples_for_h(kg_dict, h, 1)[0] +
                               self.sample_pos_triples_for_h(kg_dict, h, 1)[1] +
                               self.sample_neg_triples_for_h(kg_dict, h, self.sample_pos_triples_for_h(kg_dict, h, 1)[0], 1, highest_neg_idx)
                               for h in batch_head]
        batch_relation_tail = np.asarray(batch_relation_tail)
        batch_relation = batch_relation_tail[:, 0]
        batch_pos_tail = batch_relation_tail[:, 1]
        batch_neg_tail = batch_relation_tail[:, 2]
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embed_dim
        assert self.item_pre_embed.shape[1] == self.args.embed_dim

