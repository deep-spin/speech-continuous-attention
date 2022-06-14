base_str = """
rm -rf results/urbansound8k_acrnn_continuous_{}_discrete_attn_fused_nornn/fold_{}
python3 urbansound8k_train.py hparams/urbansound8k_acrnn.yaml \\
--train_fold_nums=[{}] \\
--valid_fold_nums=[{}] \\
--test_fold_nums=[{}] \\
--attn_domain continuous \\
--attn_max_activation {} \\
--attn_cont_encoder discrete_attn \\
--output_folder=results/urbansound8k_acrnn_continuous_{}_discrete_attn_fused_nornn/fold_{} \\
--device=cuda:2
"""


if __name__ == '__main__':
    # activations = ['softmax', 'sparsemax', 'biweight', 'triweight']
    # activations = ['softmax', 'biweight']
    activations = ['sparsemax', 'triweight']
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    folds_to_str = lambda v: ', '.join([str(x) for x in v])
    for current_fold in folds:
        train_folds = list(range(1, 11))
        del train_folds[current_fold - 1]
        print('# Fold {}'.format(current_fold))
        for act in activations:
            print(base_str.format(act, current_fold, folds_to_str(train_folds), current_fold, current_fold,
                                  act, act, current_fold))
        print('\n\n')
