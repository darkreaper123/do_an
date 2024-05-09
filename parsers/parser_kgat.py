import argparse


def parse_kgat_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")

    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='amazon-book',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=64,
                        help='Relation Embedding size.')

    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=30,
                        help='Number of epoch for early stopping')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')
    parser.add_argument('--hard_negative_sampling', type=int, default = 0)
    parser.add_argument('--one_or_two', type=int, default = 1)
    parser.add_argument('--dropout_net', type=int, default = 1)
    parser.add_argument('--data_centric', type=int, default = 0)
    parser.add_argument('--train_kg', type=int, default = 1)
    parser.add_argument('--train_cf', type=int, default = 1)
    parser.add_argument('--masking_train', type=int, default = 1)
    parser.add_argument('--n_clusters', type=int, default = 1000)
    parser.add_argument('--n_samples', type=int, default = 10)
    parser.add_argument('--full_sampling', type=int, default = 0)
    #parser.add_argument('--test_cold', type=int, default = 1) n_interactions
    parser.add_argument('--n_interactions', type=int, default = 0)
    parser.add_argument('--downstream_task_id', type=int, default = 1)
    parser.add_argument('--pretrain_task_id', type=int, default = 0)
    parser.add_argument('--n_pretrain_epochs', type=int, default = 1000)
    parser.add_argument('--n_downstream_epochs', type=int, default = 1000)
    parser.add_argument('--task_n_epochs', type=str, default = "")
    parser.add_argument('--task_ids', type=str, default = "")
    parser.add_argument('--device', default = 'cuda')
    parser.add_argument('--limit', type=float, default = 0.5)
    parser.add_argument('--test_new', type=int, default = 0)
    parser.add_argument('--check_multi_task', type=int, default = 1)
    parser.add_argument('--limits', type=str, default = "")
    parser.add_argument('--weight_loss', type=float, default = None)
    parser.add_argument('--use_two_loss_checkpoint', type=int, default = 15)
    parser.add_argument('--sigmoid_activation_before_similarity', type=int, default = 0)
    parser.add_argument('--weight_n_cf_train', type=float, default = 1.)
    parser.add_argument('--soft_optimizer', type=int, default = 0)
    parser.add_argument('--use_task_mask', type=int, default = 1)
    #
    parser.add_argument('--training_embed_weight', type=int, default = 0)
    parser.add_argument('--use_train_data_to_train_weight_embed', type=int, default = 0)
    parser.add_argument('--use_weight_embeds_no_grad', type = int, default = 0)
    parser.add_argument('--stop_training_embed_weight_epoch', type=int, default = 21)
    parser.add_argument('--use_weight_embeds_epoch_use', type = int, default = 1)
    parser.add_argument('--just_use_features_test_cold_start', type = int, default = 0)
    parser.add_argument('--ensemble_embed_features', type = int, default = 1)
    parser.add_argument('--train_weight_embed_lr', type = float, default = 1e-3)
    parser.add_argument('--use_binary_weight', type = int, default = 0)
    parser.add_argument('--meta_learning_test', type = int, default = 0)
    parser.add_argument('--training_embed_weight_batch_size', type = int, default = 200)
    parser.add_argument('--entities_embedding_used_for_second_task', type = int, default = 0)
    parser.add_argument('--n_iters_training_second_task_each_epoch', type = int, default = 3)
    parser.add_argument('--true_dropout_net', type = int, default = 0)
    #
    parser.add_argument('--n_task_masks', type = int, default = 10)
    parser.add_argument('--start_epoch_for_multi_masks_each_task_training', type = int, default = 120)
    parser.add_argument('--n_epoch_each_frame_add_one_more_mask', type = int, default = 10)
    parser.add_argument('--k_mean_for_clusters', type = int, default = 1)
    parser.add_argument('--use_first_task_mask_as_pretrained', type = int, default = 1)
    parser.add_argument('--folder_save_model', type = str, default = 'KGAT')
    parser.add_argument('--update_attention', type = int, default = 1)
    parser.add_argument('--use_task_mask_for_gradient_protecting', type = int, default = 1)
    parser.add_argument('--epoch_not_binary_mask', type = int, default = 2)
    parser.add_argument('--using_val_data_for_evaluating_mask_performance', type = int, default = 0)
    parser.add_argument('--just_one_task_mask', type = int, default = 0)
    parser.add_argument('--mf', type = int, default = 0)
    parser.add_argument('--just_one_task_mask_for_users', type = int, default = 0)
    parser.add_argument('--just_one_task_mask_for_items', type = int, default = 0)
    parser.add_argument('--ensemble_learning', type = int, default = 1)
    parser.add_argument('--one_task_mask_epoch_frame', type = int, default = 1000)
    
    
    #
    #bias reducing
    parser.add_argument('--check_using_bias_reducing_techinique', type = int, default = 1)
    parser.add_argument('--n_items_used_each_user_phase_2', type = int, default = 1)
    parser.add_argument('--alpha_for_reranking_loss', type = float, default = 1.0)
    parser.add_argument('--weight_fixing', type = float, default = 0.5)
    parser.add_argument('--n_epochs_training_reranking_loss', type = int, default = 5)
    parser.add_argument('--training_reranking_loss_every', type = int, default = 5)
    parser.add_argument('--reranking_loss_lr', type = float, default = 1e-4)
    parser.add_argument('--using_val_for_training_reranking_loss', type = int, default = 1)
    
    #upgrade GIFT4Rec
    parser.add_argument('--using_side_information_for_warm_start_users', type = int, default = 0)
    parser.add_argument('--use_embedding_mask_layer_GIFT4Rec', type = int, default = 1)
    parser.add_argument('--use_weight_embeds', type=int, default = 1)
    parser.add_argument('--training_seperately', type=int, default = 1)
    parser.add_argument('--recalculating_weight_of_neighbors', type=int, default = 1)
    
    #update UFO_SPACE
    parser.add_argument('--epoch_start_using_mask', type=int, default = 1)
    args = parser.parse_args()

    save_dir = 'trained_model/{}/{}/embed-dim{}_relation-dim{}_{}_{}_{}_lr{}_pretrain{}/'.format(
        args.folder_save_model, args.data_name, args.embed_dim, args.relation_dim, args.laplacian_type, args.aggregation_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args


