import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim

from model.GIFT4Rec_advanced.GIFT4Rec_simple_kgat import GIFT4Rec_simple_kgat
from parsers.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT
from torch.nn import Linear
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from mask_optimization_gift4u import DistributedAdamMultiTasks
sys.path.append("/content/drive/MyDrive/KGAT-pytorch-master")
from torch.utils.data import Dataset, sampler, DataLoader
class SecondTaskDataset(Dataset):
    def __init__(self, user_ids, labels):
        self.users_ids = user_ids
        self.labels = labels
    def __len__(self):
        return self.users_ids.shape[0]
    def __getitem__(self, idx):
        return self.users_ids[idx], self.labels[idx]
def evaluate(model, dataloader, Ks, device, test_cold, masking_train = 1):
    
    
    train_user_dict = dataloader.train_user_dict
    if test_cold == 0:
        #model.check_test_warm = True
        test_batch_size = dataloader.test_batch_size
        test_user_dict = dataloader.test_user_dict
    elif test_cold == 2:
        test_batch_size = dataloader.test_warm_batch_size
        test_user_dict = dataloader.test_warm_user_dict
    else:
        model.check_test_cold = True
        test_batch_size = dataloader.test_cold_batch_size
        test_user_dict = dataloader.test_cold_user_dict
        

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items + 1, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)
            #if test_cold == 1:
                #print(batch_user_ids[batch_user_ids == 0])
                #print('loz')
            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)
            batch_scores = batch_scores.cpu()
            if masking_train == 1:
                batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
            else:
                batch_metrics = calc_metrics_at_k_without_masking(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks, dataloader.train_data_masking_test_user_dict)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    model.check_test_cold = False
    model.check_test_warm = False
    return cf_scores, metrics_dict
    

def training_weight_embed(args, model, dataloader, optimizer, device, masking_train, split = 1):
    
    train_user_dict = dataloader.train_user_dict
    test_batch_size = args.training_embed_weight_batch_size
    test_user_dict = dataloader.val_user_dict.copy()
    if args.use_train_data_to_train_weight_embed:
        new_test_user_dict = train_user_dict.copy()
        for key in list(set(list(test_user_dict.keys()) + list(new_test_user_dict.keys()))):
            try:
                try:
                    new_test_user_dict[key] = np.asarray(list(new_test_user_dict[key]) + list(test_user_dict[key]))
                except:
                    new_test_user_dict[key] = test_user_dict[key]
            except:
                continue
        test_user_dict = new_test_user_dict.copy()
        masking_train = 0
        
    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores_features = []
    cf_scores_embeds = []
    #metric_names = ['precision', 'recall', 'ndcg']
    metric_names = ['precision', 'recall']
    splits = [float(i * n_items) / 10. for i in range(1, 11)]
    metrics_dict = {k: {m: [] for m in metric_names} for k in splits}
    full_users = None
    full_labels = None
    count_conflict = 0
    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        full_labels = None
        full_pred_labels = None
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)
            #print(batch_user_ids.shape)
            if True:
              if not args.entities_embedding_used_for_second_task:
                  with torch.no_grad():
                      batch_scores_features, batch_scores_embeds = model.calc_score_training_weight_embed(batch_user_ids, item_ids)       # (n_batch_users, n_items)
              else:
                  batch_scores_features, batch_scores_embeds = model.calc_score_training_weight_embed(batch_user_ids, item_ids)  
              #new      
              if not args.entities_embedding_used_for_second_task:
                  with torch.no_grad():
                      batch_scores_6_embed_4_features, batch_scores_4_embed_6_features = model.calc_score_training_weight_embed(batch_user_ids, item_ids, weights = [0.6, 0.4])       # (n_batch_users, n_items)
              else:
                   batch_scores_full = model.calc_score_training_weight_embed(batch_user_ids, item_ids, weights = [0.6, 0.4])
                   batch_scores_6_embed_4_features = batch_scores_full[0]
                   batch_scores_4_embed_6_features = batch_scores_full[1]
              #new  

              batch_scores_features = batch_scores_features.cpu()
              if masking_train == 1:
                  batch_metrics_features = calc_metrics_at_k(batch_scores_features, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), [int(split * n_items)])
              else:
                  batch_metrics_features = calc_metrics_at_k_without_masking(batch_scores_features, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), [int(split * n_items)], dataloader.train_data_masking_test_user_dict)
              batch_scores_embeds = batch_scores_embeds.cpu()
              if masking_train == 1:
                  batch_metrics_embeds = calc_metrics_at_k(batch_scores_embeds, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), [int(split * n_items)])
              else:
                  batch_metrics_embeds = calc_metrics_at_k_without_masking(batch_scores_embeds, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), [int(split * n_items)], dataloader.train_data_masking_test_user_dict) 
                    
              #new
              batch_scores_6_embed_4_features = batch_scores_6_embed_4_features.cpu()
              if masking_train == 1:
                  batch_metrics_6_embed_4_features = calc_metrics_at_k(batch_scores_6_embed_4_features, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), [int(split * n_items)])
              else:
                  batch_metrics_6_embed_4_features = calc_metrics_at_k_without_masking(batch_scores_6_embed_4_features, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), [int(split * n_items)], dataloader.train_data_masking_test_user_dict)
              batch_scores_4_embed_6_features = batch_scores_4_embed_6_features.cpu()
              if masking_train == 1:
                  batch_metrics_4_embed_6_features = calc_metrics_at_k(batch_scores_4_embed_6_features, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), [int(split * n_items)])
              else:
                  batch_metrics_4_embed_6_features = calc_metrics_at_k_without_masking(batch_scores_4_embed_6_features, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), [int(split * n_items)], dataloader.train_data_masking_test_user_dict) 
                
              #new
            
            
            
            
              batch_metric_labels = []
              count_full_metric_embeds = []
              count_full_metric_features = [] 
                
              #new
              count_full_metric_6_embed_4_features = []
              count_full_metric_4_embed_6_features = [] 
            
            
              #new
              #print(batch_metrics_embeds)
              for k in [int(split * n_items)]:
                  for m in metric_names:
                      count_full_metric_embeds.append(batch_metrics_embeds[k][m])
                      count_full_metric_features.append(batch_metrics_features[k][m])
                      count_full_metric_6_embed_4_features.append(batch_metrics_6_embed_4_features[k][m])
                      count_full_metric_4_embed_6_features.append(batch_metrics_4_embed_6_features[k][m])
            count_full_metric_embeds = np.asarray(count_full_metric_embeds).mean(axis = 0)
            count_full_metric_features = np.asarray(count_full_metric_features).mean(axis = 0)
            count_full_metric = np.hstack((count_full_metric_embeds.reshape(-1, 1),
                                           count_full_metric_features.reshape(-1, 1)))
            #print(count_full_metric)
            #print(count_full_metric[count_full_metric[:, 1] == count_full_metric[:, 0]])
            count_full_metric = np.argmax(count_full_metric, axis = 1)
            
            count_full_metric_6_embed_4_features = np.asarray(count_full_metric_6_embed_4_features).mean(axis = 0)
            count_full_metric_4_embed_6_features = np.asarray(count_full_metric_4_embed_6_features).mean(axis = 0)
            count_full_metric_2 = np.hstack((count_full_metric_6_embed_4_features.reshape(-1, 1),
                                             count_full_metric_6_embed_4_features.reshape(-1, 1)))
            count_full_metric_2 = np.argmax(count_full_metric_2, axis = 1)
            
            count_conflict += count_full_metric_2[count_full_metric_2 != count_full_metric].shape[0]
            
            
            try:
                full_users = np.concatenate((full_users, batch_user_ids.detach().cpu().numpy()))
                full_labels = np.concatenate((full_labels, count_full_metric))
            except:
                full_users = batch_user_ids.detach().cpu().numpy()
                full_labels = count_full_metric
    if full_labels[full_labels == 1].shape[0]/full_labels.shape[0] >= 0.95 or full_labels[full_labels == 1].shape[0]/full_labels.shape[0] <= 0.05:
        return 
    second_task_dataset = SecondTaskDataset(full_users.reshape(-1, 1), full_labels.reshape(-1, 1))
    classes_weights = [full_labels[full_labels == 1].shape[0], full_labels[full_labels == 0].shape[0]]
    classes_weights = np.asarray(classes_weights)
    classes_weights = 1./classes_weights
    classes_weights = classes_weights[full_labels]
    classes_weights = torch.tensor(classes_weights).double()
    #classes_weights = classes_weights/torch.sum(classes_weights)
    second_task_sampler = sampler.WeightedRandomSampler(classes_weights, len(second_task_dataset), replacement=True)
    second_task_loader = DataLoader(second_task_dataset, batch_size = test_batch_size, sampler = second_task_sampler)
    full_true_labels = None
    full_pred_labels = None
    n_epochs = args.n_iters_training_second_task_each_epoch
    for epoch in range(1, n_epochs + 1):
        for idx, batch in enumerate(second_task_loader):
            batch_user_ids, batch_labels = batch
            batch_user_ids = torch.LongTensor(batch_user_ids.reshape(-1)).to(device)
            batch_labels = torch.LongTensor(batch_labels.reshape(-1)).to(device)
            if True:
                batch_pred_labels = model.calc_cf_full_weight_averaged(batch_user_ids)
                #print(pred_labels.shape)
                label_one_weight = batch_labels[batch_labels == 1].shape[0]/batch_labels.shape[0]
                #print(label_one_weight)
                #print(label_one_weight)
                loss = nn.CrossEntropyLoss(weight = torch.tensor([label_one_weight, 1 - label_one_weight]).to(device))(batch_pred_labels, batch_labels)
                optimizer.zero_grad()
                loss.backward() 
                if not args.use_weight_embeds_no_grad:
                    id_to_mask_params = get_dict_id_params(model, 1)
                    optimizer.step(id_to_mask_params = id_to_mask_params)
                else:
                    optimizer.step()
                pred_labels = batch_pred_labels.detach().cpu().numpy()
                pred_labels = np.argmax(pred_labels, axis = 1)
                labels = batch_labels.detach().cpu().numpy()
                if epoch == n_epochs:
                    try:
                      full_true_labels = labels
                      full_pred_labels = pred_labels
                    except:
                      full_true_labels = np.concatenate((full_true_labels, labels))
                      full_pred_labels = np.concatenate((full_pred_labels, pred_labels))
    print(confusion_matrix(full_true_labels, full_pred_labels, normalize = 'true'))
                
def get_dict_id_params(model, task_id):
    true_task_id = (task_id + 1) % 2
    #print(true_task_id)
    source_dict = {}
    save_index_weight = None
    save_index_bias = None
    for index, (name, params) in enumerate(model.named_parameters()):
        if name.split('.')[-1] == 'weight':
            save_index_weight = index
        if name.split('.')[-1] == 'bias':
            save_index_bias = index
        if 'mask_weight_{}'.format(true_task_id) in name:
            if save_index_weight not in source_dict:
                source_dict[save_index_weight] = params.data
        if 'mask_bias_{}'.format(true_task_id) in name:
            if save_index_bias not in source_dict:
                source_dict[save_index_bias] = params.data
    return source_dict   
    
def train(args):
    # seed
    if args.data_name != 'douban_one_task':
        args.seed = random.randint(0, int(1e9) + 1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)
    # GPU / CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    # load data
    data = DataLoaderKGAT(args, logging)
    if True:
        if False:
            for user, items in data.val_user_dict.items():
                if user not in data.train_user_dict:
                    data.train_user_dict[user] = items
                else:
                    data.train_user_dict[user] = np.concatenate((data.train_user_dict[user], items))
            data.val_user_dict = {user:items[-1:] for user, items in data.train_user_dict.items()}
            data.train_user_dict = {user:items[:-1] for user, items in data.train_user_dict.items()}
        data.test_user_dict.update(data.test_warm_user_dict)
    #data.device = device
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None
    print(data.max_id)
    # construct model & optimizer
    n_items = data.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)
    model = GIFT4Rec_simple_kgat(args, data.n_users, data.n_entities, data.n_relations,
                                 data.user_category_features, data.user_numeric_features, 
                                 data.item_category_features, data.item_numeric_features,
                                 data.max_size_embedding_category, device,
                                 data.n_category, data.n_numeric,
                                 data.max_id,
                                 item_ids,
                                 data.A_in, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)
    if not args.use_weight_embeds_no_grad:
        print(args.use_weight_embeds_no_grad)
        cf_optimizer = DistributedAdamMultiTasks(model.parameters(), lr=args.lr)
        training_embed_optimizer = DistributedAdamMultiTasks(model.parameters(), lr= args.train_weight_embed_lr)
    else:
        cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
        training_embed_optimizer = optim.Adam(model.parameters(), lr= args.train_weight_embed_lr)
    #if data.data_name == 'ml_100k_one_task' and args.stop_training_embed_weight_epoch > 9 + 1:args.stop_training_embed_weight_epoch = 9 + 1
    model.to(device)
    logging.info(model)

    

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = args.Ks.replace("'", "")
    Ks = Ks.replace("[", "")
    Ks = Ks.replace("]", "")
    Ks = [int(elem) for elem in Ks.split(',')]
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}
    model.check_2 = False
    # train model
    new_lr = args.lr
    model.hard_or_random = args.hard_negative_sampling
    old_lr = args.lr
    for epoch in range(1, args.n_epoch + 1):
        model.during_testing = False
        #new
        model.ensemble_embed_features = args.ensemble_embed_features
        #new
        time0 = time()
        if args.use_weight_embeds:
            if epoch == args.use_weight_embeds_epoch_use:
                model.use_weight_embeds = args.use_weight_embeds
        model.check = args.dropout_net
        #new
        model.use_binary_weight = False
        #new
        model.train()
        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        if args.data_name == 'ml_1m_one_task':n_cf_batch = 10
        if args.data_name == 'douban_one_task':n_cf_batch = 10
        check_exit = False
        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            #print(cf_batch_user.shape)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')
            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                #sys.exit()
                check_exit = True
                break
            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training with raw embed: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))
        if check_exit:
            break
        
        for iter in range(1, n_cf_batch + 1):
            model.using_features_one_time = True
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            #print(cf_batch_user.shape)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')
            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                #sys.exit()
                check_exit = True
                break
            cf_batch_loss.backward()
            if not args.use_weight_embeds_no_grad and epoch > 1:
                id_to_mask_params = get_dict_id_params(model, 0)
                #print(id_to_mask_params)
                cf_optimizer.step(id_to_mask_params = id_to_mask_params)
            else:
                cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()
        
            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training with user features: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
            model.using_features_one_time = False
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))
        if check_exit:
            break    
        
        
        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model.check = False
        if not args.recalculating_weight_of_neighbors:
            model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))
        logging.info('CF Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))
        #print(model.check)
        # evaluate cf
        model.use_binary_weight = args.use_binary_weight
        if not args.ensemble_embed_features:
            model.ensemble_embed_features = True
        model.during_testing = True
        model.using_features_one_time = True
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            _, metrics_dict = evaluate(model, data, Ks, device, 0, args.masking_train)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])

        #if args.one_or_two == 2 or args.just_use_features_test_cold_start:
        #new
        count = 1
        #model.using_features_one_time = True
        if False:
            if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
                time6 = time()
                _, metrics_dict = evaluate(model, data, Ks, device, 2)
                logging.info('CF Evaluation (Warm - start): Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                    epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
                if True:
                  count += 1
                  for k in Ks:
                    for m in ['precision', 'recall', 'ndcg']:
                        metrics_list[k][m][-1] = metrics_dict[k][m] + metrics_list[k][m][-1]
        #new
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            _, metrics_dict = evaluate(model, data, Ks, device, 1)
            logging.info('CF Evaluation (Cold - start): Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
            if True:
              count += 1
              for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m][-1] = metrics_dict[k][m] + metrics_list[k][m][-1]
        for k in Ks:
          for m in ['precision', 'recall', 'ndcg']:
            metrics_list[k][m][-1] /= float(count)
        model.during_testing = False
        model.using_features_one_time = False
        if args.training_embed_weight and epoch + 1 < args.stop_training_embed_weight_epoch:
          #new
          #new
          if args.use_train_data_to_train_weight_embed and epoch >= 16:
              args.training_embed_weight = 0
          model.using_features_one_time = True
          training_weight_embed(args, model, data, training_embed_optimizer, device, args.masking_train)
          model.during_testing = True
          if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
              #new
              model.ensemble_embed_features = True
              #new
              time7 = time()
              _, metrics_dict = evaluate(model, data, Ks, device, 0, args.masking_train)
              logging.info('CF Evaluation after training weight embed: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                  epoch, time() - time7, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
              if False:
                  time7 = time()
                  #new
                  _, metrics_dict_warm = evaluate(model, data, Ks, device, 2)
                  logging.info('CF Evaluation after training weight embed (Warm - start): Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                      epoch, time() - time7, metrics_dict_warm[k_min]['precision'], metrics_dict_warm[k_max]['precision'], metrics_dict_warm[k_min]['recall'], metrics_dict_warm[k_max]['recall'], metrics_dict_warm[k_min]['ndcg'], metrics_dict_warm[k_max]['ndcg']))
              model.ensemble_embed_features = False
              #new
              _, metrics_dict_cold = evaluate(model, data, Ks, device, 1)
              logging.info('CF Evaluation after training weight embed (Cold - start): Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                  epoch, time() - time7, metrics_dict_cold[k_min]['precision'], metrics_dict_cold[k_max]['precision'], metrics_dict_cold[k_min]['recall'], metrics_dict_cold[k_max]['recall'], metrics_dict_cold[k_min]['ndcg'], metrics_dict_cold[k_max]['ndcg']))
              #count = 3.
              count = 2.
              #if count * metrics_list[k_max]['recall'][-1] < (metrics_dict_cold[k_max]['recall'] + metrics_dict[k_max]['recall'] + metrics_dict_warm[k_max]['recall']):
              #if count * metrics_list[k_max]['recall'][-1] < (metrics_dict_cold[k_max]['recall'] + metrics_dict[k_max]['recall'] + metrics_dict_warm[k_max]['recall']):
              if count * metrics_list[k_max]['recall'][-1] < (metrics_dict_cold[k_max]['recall'] + metrics_dict[k_max]['recall']):
                for k in Ks:
                    for m in ['precision', 'recall', 'ndcg']:
                        metrics_list[k][m][-1] = (metrics_dict[k][m] + metrics_dict_cold[k][m]) / count
          model.using_features_one_time = False
        else:
          args.training_embed_weight = False
        time8 = time()
        logging.info('Averaged CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(epoch, time() - time8, metrics_list[k_min]['precision'][-1], metrics_list[k_max]['precision'][-1], metrics_list[k_min]['recall'][-1], metrics_list[k_max]['recall'][-1], metrics_list[k_min]['ndcg'][-1], metrics_list[k_max]['ndcg'][-1]))
        if True:
            if True:
              #best_recall, should_stop = early_stopping(metrics_list[k_max]['recall'], args.stopping_steps)
              best_ndcg, should_stop = early_stopping(metrics_list[k_max]['recall'], args.stopping_steps)
              #if metrics_list[k_max]['recall'].index(best_recall) == len(epoch_list) - 1:
              if metrics_list[k_max]['recall'].index(best_ndcg) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

              if should_stop:
                #break
                pass
        if not args.ensemble_embed_features:
            model.ensemble_embed_features = False   
              
    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))


def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderKGAT(args, logging)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))



if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)
    # predict(args)


