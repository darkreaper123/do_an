import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim

from model.KGAT_2_raw_new_ver import KGAT_2_raw_new_ver
from parsers.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT
from torch.nn import Linear
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
sys.path.append("/content/drive/MyDrive/KGAT-pytorch-master")
def evaluate(model, dataloader, Ks, device, test_cold, masking_train = 1):
    
    train_user_dict = dataloader.train_user_dict
    if test_cold == 0:
        test_batch_size = dataloader.test_batch_size
        test_user_dict = dataloader.test_user_dict
    elif test_cold == 2:
        test_batch_size = dataloader.test_warm_batch_size
        test_user_dict = dataloader.test_warm_user_dict
    else:
        test_batch_size = dataloader.test_cold_batch_size
        test_user_dict = dataloader.test_cold_user_dict
        model.check_test_cold = True

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
    return cf_scores, metrics_dict
    

def training_weight_embed(args, model, dataloader, optimizer, device, masking_train, split = 1):
    
    train_user_dict = dataloader.train_user_dict
    test_batch_size = 100
    test_user_dict = dataloader.val_user_dict.copy()
    if args.use_train_data_to_train_weight_embed:
        new_test_user_dict = train_user_dict.copy()
        for key in list(set(list(test_user_dict.keys()) + list(new_test_user_dict.keys()))):
            try:
                new_test_user_dict[key] = np.asarray(list(new_test_user_dict[key]) + list(test_user_dict[key]))
            except:
                new_test_user_dict[key] = test_user_dict[key]
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
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in [int(split * n_items)]}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        full_labels = None
        full_pred_labels = None
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)
            #print(batch_user_ids.shape)
            if True:
              with torch.no_grad():
                  batch_scores_features, batch_scores_embeds = model.calc_score_training_weight_embed(batch_user_ids, item_ids)       # (n_batch_users, n_items)

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
              batch_metric_labels = []
              count_full_metric_embeds = []
              count_full_metric_features = [] 
              #print(batch_metrics_embeds)
              for k in [int(split * n_items)]:
                  for m in metric_names:
                      count_full_metric_embeds.append(batch_metrics_embeds[k][m])
                      count_full_metric_features.append(batch_metrics_features[k][m])
            count_full_metric_embeds = np.asarray(count_full_metric_embeds).mean(axis = 0)
            count_full_metric_features = np.asarray(count_full_metric_features).mean(axis = 0)
            count_full_metric = np.hstack((count_full_metric_embeds.reshape(-1, 1),
                                           count_full_metric_features.reshape(-1, 1)))
            #print(count_full_metric[count_full_metric[:, 1] == count_full_metric[:, 0]])
            count_full_metric = np.argmax(count_full_metric, axis = 1)
            labels = torch.LongTensor(count_full_metric).to(device)
            #print(labels.shape)
            pred_labels = model.calc_cf_full_weight_averaged(batch_user_ids)
            #print(pred_labels.shape)
            label_one_weight = labels[labels == 1].shape[0]/labels.shape[0]
            #print(label_one_weight)
            loss = nn.CrossEntropyLoss(weight = torch.tensor([label_one_weight, 1 - label_one_weight]).to(device))(pred_labels, labels)
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
            pred_labels = pred_labels.detach().cpu().numpy()
            pred_labels = np.argmax(pred_labels, axis = 1)
            labels = labels.detach().cpu().numpy()
            try:
              full_labels = labels
              full_pred_labels = pred_labels
            except:
              full_labels = np.concatenate((full_labels, labels))
              full_pred_labels = np.concatenate((full_pred_labels, pred_labels))
        print(confusion_matrix(full_labels, full_pred_labels, normalize = 'true'))

def train(args):
    # seed
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
    model = KGAT_2_raw_new_ver(args, data.n_users, data.n_entities, data.n_relations,
                               data.user_category_features, data.user_numeric_features, 
                               data.item_category_features, data.item_numeric_features,
                               data.max_size_embedding_category, device,
                               data.n_category, data.n_numeric,
                               data.max_id,
                               item_ids,
                               data.A_in, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)
    #if data.data_name == 'ml_100k_one_task' and args.stop_training_embed_weight_epoch > 9 + 1:args.stop_training_embed_weight_epoch = 9 + 1
    model.to(device)
    logging.info(model)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    training_embed_optimizer = optim.Adam(model.parameters(), lr= args.train_weight_embed_lr)

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
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()
        if (epoch + 1) % 10 == 0:
            #new_lr = new_lr * 0.1
            new_lr = new_lr * 1.
            cf_optimizer = optim.Adam(model.parameters(), lr=new_lr)
            kg_optimizer = optim.Adam(model.parameters(), lr=new_lr)
        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        if args.data_name == 'ml_1m_one_task':n_cf_batch = 10
        if args.data_name == 'douban_one_task':n_cf_batch = 10
        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')
            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()
            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
        if args.data_name == 'ml_1m_one_task':n_kg_batch = 10
        if args.data_name == 'douban_one_task':n_kg_batch = 10
        for iter in range(1, n_kg_batch + 1):
            if not args.train_kg:
                break
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

        # update attention
        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model.check = False
        model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))
        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))
        #print(model.check)
        # evaluate cf
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
        time8 = time()
        logging.info('Averaged CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(epoch, time() - time8, metrics_list[k_min]['precision'][-1], metrics_list[k_max]['precision'][-1], metrics_list[k_min]['recall'][-1], metrics_list[k_max]['recall'][-1], metrics_list[k_min]['ndcg'][-1], metrics_list[k_max]['ndcg'][-1]))
        if True:
            if True:
              #best_recall, should_stop = early_stopping(metrics_list[k_max]['recall'], args.stopping_steps)
              best_recall, should_stop = early_stopping(metrics_list[k_max]['recall'], args.stopping_steps)
              #if metrics_list[k_max]['recall'].index(best_recall) == len(epoch_list) - 1:
              if metrics_list[k_max]['recall'].index(best_recall) == len(epoch_list) - 1:
                #save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

            if should_stop:
                #break
                pass
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


