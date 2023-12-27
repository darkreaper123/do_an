import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from model.Melu.embeddings import user, item

torch.autograd.set_detect_anomaly(True)

class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.item_emb = item(config)
        self.user_emb = user(config)
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    def forward(self, x, training = True, local_update = False):
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10190], requires_grad=False)
        gender_idx = Variable(x[:, 10190], requires_grad=False)
        age_idx = Variable(x[:, 10191], requires_grad=False)
        occupation_idx = Variable(x[:, 10192], requires_grad=False)
        area_idx = Variable(x[:, 10193], requires_grad=False)
        if local_update:
            with torch.no_grad():
                item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
                user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        else:
            item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
            user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.linear_out(x)


class MeLU(torch.nn.Module):
    def __init__(self, config):
        super(MeLU, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = user_preference_estimator(config)
        self.local_model = user_preference_estimator(config)
        self.local_lr = config['local_lr']
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.local_optim = torch.optim.Adam(self.local_model.parameters(), lr=self.local_lr)
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        if True:
            for idx in range(num_local_update):
                support_set_y_pred = self.local_model(support_set_x)
                loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
                self.local_model.zero_grad()
                loss.backward()
                self.local_optim.step()
        query_set_y_pred = self.local_model(query_set_x, local_update = True)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        self.meta_optim.zero_grad()
        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            loss_q.backward()
            count = 0
            for parameters in self.model.parameters():
                count_j  = 0 
                for local_parameters in self.local_model.parameters():
                    if count_j < count:
                        count_j += 1
                    else:
                        parameters.grad = local_parameters.grad
                        break
                count += 1
            self.local_model.load_state_dict(self.keep_weight)
        for parameters in self.model.parameters():
            parameters.grad = parameters.grad/float(batch_sz)
        
        #losses_q = torch.stack(losses_q).mean(0)
        #losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        self.local_model.load_state_dict(deepcopy(self.model.state_dict()))
        return
