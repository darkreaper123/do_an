import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear
def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

import math
from torch.nn import init
from torch.nn import Parameter
class MaskLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, n_tasks = 2):
        super().__init__(in_features = in_features, out_features = out_features, bias = bias,
                 device=None, dtype=None)
        self.n_tasks = n_tasks
        self.check_bias = bias
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mask_weights = nn.ParameterList([nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs)) for i in range(n_tasks)])
        for i in range(n_tasks):
            self.register_parameter('mask_weight_{}'.format(i), self.mask_weights[i])
        if bias:
            self.mask_biases =  nn.ParameterList([nn.Parameter(torch.empty(out_features, **factory_kwargs)) for i in range(n_tasks)])
            for i in range(n_tasks):
                self.register_parameter('mask_bias_{}'.format(i), self.mask_biases[i])

        self.reset_mask_parameters()  
    def reset_mask_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for index in range(self.n_tasks):
            init.kaiming_uniform_(self.mask_weights[index], a=math.sqrt(5))
            if self.check_bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.mask_weights[index])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.mask_biases[index], -bound, bound)
    def forward(self, input, index = None):
        if index is not None:
            #if index == 1:
                #print(torch.sigmoid(self.mask_weights[1]))
                #print(torch.sigmoid(self.mask_weights[0]))
            try:
                return F.linear(input, self.weight * torch.sigmoid(self.mask_weights[index]), self.bias * torch.sigmoid(self.mask_biases[index]))
            except:
                return F.linear(input, self.weight * torch.sigmoid(self.mask_weights[index]), None)
        else:
            return F.linear(input, self.weight, self.bias)


class GIFT4Rec_advanced_mf(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 user_category_features, user_numeric_features,
                 item_category_features, item_numeric_features, 
                 max_size_category_embedding, device,
                 n_category, n_numeric,
                 max_id,
                 items_id,
                 A_in=None,
                 user_pre_embed=None, item_pre_embed=None):

        super(GIFT4Rec_advanced_mf, self).__init__()
        self.use_pretrain = args.use_pretrain
        self.use_weight_embeds_no_grad = args.use_weight_embeds_no_grad
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.use_weight_embeds = False
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.device = device
        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim, padding_idx = 0)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
        
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        sum_n_embed = self.embed_dim

        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False
        try:
            self.user_category_features = torch.from_numpy(user_category_features.astype(int))
        except:
            self.user_category_features = None
        try:
            self.user_numeric_features = torch.from_numpy(user_numeric_features.astype(np.float32))
        except:
            self.user_numeric_features = None
        try:
            self.item_category_features = torch.from_numpy(item_category_features.astype(int))
        except:
            self.item_category_features = None
        try:
            self.item_numeric_features = torch.from_numpy(item_numeric_features.astype(np.float32))
        except:
            self.item_numeric_features = None
        self.max_size_category_embedding = max_size_category_embedding
        self.embedding_layer_category = nn.ModuleList([nn.Embedding(max_size_category_embedding[i] + 1, sum_n_embed, padding_idx = 0) 
                                                       for i in range(2)])
        self.n_numeric = n_numeric
        self.n_category = n_category
        self.check = True
        self.check_2 = True
        self.n_embedding = sum_n_embed
        self.dense_numeric_layers = nn.ModuleList([Linear(n_numeric[i], sum_n_embed) for i in range(2)])
        n_hiddens = sum_n_embed * 2
        if self.use_weight_embeds_no_grad:
            self.weight_averaged_features_layer = nn.ModuleList([Linear(2 * sum_n_embed, n_hiddens),
                                                                 Linear(n_hiddens, n_hiddens),
                                                                 Linear(n_hiddens, 1)])
        else:
            self.weight_averaged_features_layer = nn.ModuleList([MaskLinear(2 * sum_n_embed, n_hiddens, n_tasks = 2),
                                                                 MaskLinear(n_hiddens, n_hiddens, n_tasks = 2),
                                                                 MaskLinear(n_hiddens, 1, n_tasks = 2)])
        self.max_id = max_id
        self.items_id = items_id
        self.check_3 = True
        self.check_training_weight_embed_using_features = False
        self.check_training_weight_embed_using_embeds = False
        self.check_training_weight_embed = False
        self.args = args
        self.use_binary_weight = False
        self.ensemble_embed_features = False
        self.check_print_new_ids = False
        self.check_test_cold = False
        self.during_testing = False
        self.check_test_warm = False
    def get_category_features(self, inputs_category, device, index):
        embedding_category = self.embedding_layer_category[index](inputs_category)
        n_fake_category = inputs_category.shape[-1]
        n_category_inputs = 1. / torch.sum(inputs_category > 0, axis = -1)
        n_category_inputs = torch.nan_to_num(n_category_inputs)
        n_category_inputs_full = torch.full((n_fake_category,) + n_category_inputs.shape, 0, dtype = torch.float32).to(device)
        n_category_inputs_full += n_category_inputs
        #print(n_category_inputs_full.shape)
        #print(0)
        n_category_inputs_full = n_category_inputs_full.permute(1, 0)
        n_category_inputs_full = n_category_inputs_full.to(device)
        #print(n_category_inputs_full.shape)
        #print(10)
        #print(embedding_category.shape)
        #print(100)
        embedding_category = torch.matmul(embedding_category.permute(0, 2, 1), 
                                          n_category_inputs_full.view(n_category_inputs_full.shape + (1,))).permute(0, 2, 1)
        #print(embedding_category.shape)
        return embedding_category
    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        return ego_embed
    def hard_negative_sampling(self, pos_ids):
        neg_id = 100
        return torch.from_numpy(np.argsort(self.get_similarity[pos_ids.detach().cpu().numpy()])[:, neg_id]).to(self.device)      
    def get_features_ids(self, ids, user_or_items):
        index = (user_or_items + 1)%2
        if user_or_items == 1:
          if self.n_category[0] > 0:
              inputs_category = self.user_category_features[ids.detach().cpu().numpy() - self.n_entities].to(self.device)
          if self.n_numeric[0] > 0:
              inputs_numeric = self.user_numeric_features[ids.detach().cpu().numpy() - self.n_entities].to(self.device)
        else:
          if self.n_category[1] > 0:
              inputs_category = self.item_category_features[ids.detach().cpu().numpy()].to(self.device)
          if self.n_numeric[1] > 0: 
              inputs_numeric = self.item_numeric_features[ids.detach().cpu().numpy()].to(self.device)
        features = None
        if self.n_category[index] > 0:
            embedding_category = self.get_category_features(inputs_category, self.device, index)
            embedding_category = embedding_category.view(embedding_category.shape[0], 1, embedding_category.shape[-1])
            #print(embedding_category.shape)
            #print(embedding_category[torch.isnan(embedding_category.clone().detach().cpu())])
            #print(1)
            try:
                features += embedding_category
            except:
                features = embedding_category
        if self.n_numeric[index] > 0:
            embedding_numeric = self.dense_numeric_layers[index](inputs_numeric)
            embedding_numeric = embedding_numeric.view(embedding_numeric.shape[0], 1, embedding_numeric.shape[-1])
            embedding_numeric = F.relu(embedding_numeric)
            #print(embedding_numeric.shape)
            #print(embedding_numeric[torch.isnan(embedding_numeric.clone().detach().cpu())])
            #print(2)
            try:
                features += embedding_numeric
            except:
                features = embedding_numeric
        return features
    def calc_one_embeddings(self, all_embed, ids, user_or_items, fix_weight_for_embed = None):
        #print(ids)
        index = (user_or_items + 1) % 2
        old_ids = ids.detach()
        new_ids = ids.detach().cpu().numpy()
        old_new_ids = new_ids.copy()
        embed = all_embed[new_ids]
        if user_or_items == 1:
            #if self.check_test_warm:
            if self.check_test_cold:
                embed = 0 * embed 
                embed += torch.mean(all_embed[-self.n_users:self.max_id[index] + self.n_entities], dim = 0).view(1, -1)
        def random_batch(batch_size, limit):
            random_entities = np.random.permutation(batch_size)[:int(0.2 * batch_size)]
            return torch.from_numpy(random_entities.astype(int))
        def random_batch_2(batch_size, limit):
            random = np.random.rand(batch_size, 1)
            random[random <= limit] = 0.
            random[random > limit] = 1.
            random = np.full((self.n_embedding,) + random.shape, random)
            random = np.transpose(random, (1, 2, 0))
            return torch.from_numpy(random.astype(np.float32))
        random_entities = random_batch_2(ids.shape[0], 0.5).to(self.device)
        #print(random_entities)
        embed = embed.view(ids.shape[0], 1, -1)
        if self.check and ((self.n_category[index] > 0) or (self.n_numeric[index] > 0)) and self.args.dropout_net:
            embed *= random_entities
            #print(embedding_users_numeric[torch.isnan(embedding_users_numeric.clone().detach().cpu())])
        features = self.get_features_ids(ids, user_or_items)
        #print(features[torch.isnan(features.clone().detach().cpu())])
        if features is not None:
            features /= self.n_numeric[index] + self.n_category[index]
            #print(features[torch.isnan(features.clone().detach().cpu())])
            #print(3)
            if self.ensemble_embed_features:
                  if self.use_weight_embeds:
                      concat_embed = torch.concat([embed, features], axis = -1)
                      if self.use_weight_embeds_no_grad:
                          with torch.no_grad():
                              for index, layer in enumerate(self.weight_averaged_features_layer):
                                  if index == 0:
                                      weight_averaged = layer(concat_embed)
                                      continue
                                  weight_averaged = layer(weight_averaged)
                              weight_averaged = torch.sigmoid(weight_averaged)     
                      else:
                          for index, layer in enumerate(self.weight_averaged_features_layer):
                              if index == 0:
                                  if not self.during_testing:
                                      weight_averaged = layer(concat_embed, 0)
                                  else:
                                      weight_averaged = layer(concat_embed)
                                  continue
                              if not self.during_testing:
                                  weight_averaged = layer(weight_averaged, 0)
                              else:
                                  weight_averaged = layer(weight_averaged)
                          weight_averaged = torch.sigmoid(weight_averaged)  
                      if self.use_binary_weight:
                          weight_averaged = nn.Threshold(0.5, 0.)(weight_averaged)
                          weight_averaged = nn.Threshold(-0.5, 1.)(-weight_averaged)
                      weight_averaged = torch.concat([weight_averaged, 1 - weight_averaged], dim = -1)
                      concat_embed = concat_embed.view(concat_embed.shape[0], 2, features.shape[-1])
                      concat_embed = torch.matmul(concat_embed.permute(0, 2, 1), weight_averaged.permute(0, 2, 1))
                      embed = concat_embed.permute(0, 2, 1)
            else:
                  embed = 0.5 * embed + 0.5 * features
            #print(embed[torch.isnan(embed.clone().detach().cpu())])
            #print(embed.shape)
            #print(7)
            if self.args.dropout_net:
                #embed = embed.view(-1, 1, embed.shape[-1]) 
                if self.check:
                    embed = embed * random_entities + (1 - random_entities) * features 
                else:
                    if self.check_training_weight_embed and user_or_items:
                        if fix_weight_for_embed is None:
                            if self.check_training_weight_embed_using_features:
                                return features[:, 0, :]
                            elif self.check_training_weight_embed_using_embeds:
                                return embed[:, 0, :] 
                        else:
                            return embed[:, 0, :] * fix_weight_for_embed + features[:, 0, :] * (1 - fix_weight_for_embed)
            #print(embed[torch.isnan(embed.clone().detach().cpu())])
            #print(8)
        embed = embed[:, 0, :]
        #embed = torch.nan_to_num(embed)
        return embed
    def calc_weight_averaged(self, all_embed, ids, user_or_items):
        with torch.no_grad():
          index = (user_or_items + 1) % 2
          new_ids = ids.detach().cpu().numpy()
          if user_or_items:
            new_ids[new_ids > self.max_id[index] + self.n_entities] = 0
          embed = all_embed[new_ids]
          features = self.get_features_ids(ids, user_or_items)
          #print(features[torch.isnan(features.clone().detach().cpu())])
          features /= self.n_numeric[index] + self.n_category[index]
        concat_embed = torch.concat([features[:, 0, :], embed], dim = -1)
        for index, layer in enumerate(self.weight_averaged_features_layer):
            if index == 0:
                try:
                    weight_averaged = layer(concat_embed, 1)
                except:
                    weight_averaged = layer(concat_embed)
                continue
            try:
                weight_averaged = layer(weight_averaged, 1)
            except:
                weight_averaged = layer(weight_averaged)
        weight_averaged = torch.sigmoid(weight_averaged)  
        #print(weight_averaged)   
        return weight_averaged
            
    def calc_cf_full_embeddings(self, user_ids, item_pos_ids, item_neg_ids):
                                # (n_users + n_entities, concat_dim)
        all_embed = self.calc_cf_embeddings()
        user_embed = self.calc_one_embeddings(all_embed, user_ids, 1)                             # (cf_batch_size, concat_dim)
        item_pos_embed = self.calc_one_embeddings(all_embed, item_pos_ids, 0)                     # (cf_batch_size, concat_dim)
        item_neg_embed = self.calc_one_embeddings(all_embed, item_neg_ids, 0)
        return all_embed, user_embed, item_pos_embed, item_neg_embed
    def calc_cf_full_embeddings_test(self, user_ids, item_ids, weight = None):
                                # (n_users + n_entities, concat_dim)
        all_embed = self.calc_cf_embeddings()
        #print(torch.min(item_ids))
        #print(torch.min(user_ids))
        user_embed = self.calc_one_embeddings(all_embed, user_ids, 1, fix_weight_for_embed = weight)                             # (cf_batch_size, concat_dim)
        #print(user_ids)
        item_embed = self.calc_one_embeddings(all_embed, item_ids, 0, fix_weight_for_embed = weight)                     # (cf_batch_size, concat_dim)
        
        return user_embed, item_embed
    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, hard_or_random = 0):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        # Equation (12)
        #print(user_embed.shape)
        #print(item_pos_embed.shape)
        all_embed, user_embed, item_pos_embed, item_neg_embed = self.calc_cf_full_embeddings(user_ids, item_pos_ids, item_neg_ids)
        #print(user_embed.shape)
        #print(item_neg_embed.shape)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        #print(item_pos_ids[torch.sum(torch.isnan(pos_score), dim = -1)])
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)
        #print(item_neg_ids[torch.sum(torch.isnan(neg_score), dim = -1)])
        # Equation (13)
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score + 1e-7)
        #print(cf_loss)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)                                     # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list


    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        user_embed, item_embed = self.calc_cf_full_embeddings_test(user_ids, item_ids)
        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        #print(user_ids)
        return cf_score
    def calc_score_training_weight_embed(self, user_ids, item_ids, weights = None):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        if weights is not None:
            cf_scores = []
            self.check_training_weight_embed = True
            for weight in weights:
                user_embed, item_embed = self.calc_cf_full_embeddings_test(user_ids, item_ids, weight)
                cf_scores.append(torch.matmul(user_embed, item_embed.transpose(0, 1)))
            self.check_training_weight_embed = False
            return cf_scores
        else:
            self.check_training_weight_embed = True
            self.check_training_weight_embed_using_features = True
            user_embed, item_embed = self.calc_cf_full_embeddings_test(user_ids, item_ids)
            cf_score_features = torch.matmul(user_embed, item_embed.transpose(0, 1))
            self.check_training_weight_embed_using_features = False
            self.check_training_weight_embed_using_embeds = True
            user_embed, item_embed = self.calc_cf_full_embeddings_test(user_ids, item_ids)
            cf_score_embeds = torch.matmul(user_embed, item_embed.transpose(0, 1))
            self.check_training_weight_embed_using_embeds = False
            self.check_training_weight_embed = False
            return cf_score_features, cf_score_embeds
        
            
    def calc_cf_full_weight_averaged(self, user_ids):
                                # (n_users + n_entities, concat_dim)
        with torch.no_grad():
          all_embed = self.calc_cf_embeddings()
        weight_averaged = self.calc_weight_averaged(all_embed, user_ids, 1)                             # (cf_batch_size, concat_dim)
        weight_averaged = torch.concat([weight_averaged, 1 - weight_averaged], dim = -1)
        return weight_averaged      

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)


