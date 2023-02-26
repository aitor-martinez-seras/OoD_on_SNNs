

def load_in_distribution_data(in_dataset, config, datasets_loader, datasets_path, datasets_conf, logger):
    # Get the batch size and data loaders to obtain the data splits
    batch_size = get_batch_size(config, in_dataset, logger)
    in_dataset_data_loader = datasets_loader[in_dataset](datasets_path)

    # Load both splits
    train_data = in_dataset_data_loader.load_data(
        split='train', transformation_option='test', output_shape=datasets_conf[in_dataset]['input_size'][1:]
    )
    test_data = in_dataset_data_loader.load_data(
        split='test', transformation_option='test', output_shape=datasets_conf[in_dataset]['input_size'][1:]
    )

    # Define loaders. Use a seed for train loader
    g_ind = torch.Generator()
    g_ind.manual_seed(args.ind_seed)
    train_loader = load_dataloader(train_data, batch_size, shuffle=True, generator=g_ind)
    test_loader = load_dataloader(test_data, batch_size, shuffle=True, generator=g_ind)

    # Extract useful variables for future operations
    class_names = train_data.classes
    return train_loader, test_loader, class_names


# input_size = datasets_conf[in_dataset]['input_size']
# hidden_neurons = model_archs[model_name][in_dataset][0]
# output_neurons = datasets_conf[in_dataset]['classes']

def diff_sizes(ood_dataset, datasets_loader, datasets_path, datasets_conf, in_dataset):
    if ood_dataset.split('/')[0] == 'MNIST-C':
        ood_dataset_data_loader = datasets_loader['MNIST-C'](
            datasets_path, option=ood_dataset.split('/')[1]
        )

    else:
        ood_dataset_data_loader = datasets_loader[ood_dataset](datasets_path)

    ood_data = ood_dataset_data_loader.load_data(
        split='test', transformation_option='test',
        output_shape=datasets_conf[in_dataset]['input_size'][1:]
    )

    # Define loaders. Use a seed for ood loader
    g_ood = torch.Generator()
    g_ood.manual_seed(8)
    ood_loader = load_dataloader(ood_data, batch_size, shuffle=True, generator=g_ood)

    size_test_data = len(preds_test)
    size_ood_data = len(ood_data)

    # Ensure we have same number of samples for test and ood
    if size_ood_data == size_test_data:
        pass

    elif size_ood_data < size_test_data:
        logger.info(f"Using training data as test OOD data for {ood_dataset} dataset")

        # Load the train data of OOD dataset
        ood_data = ood_dataset_data_loader.load_data(
            split='train', transformation_option='test',
            output_shape=datasets_conf[in_dataset]['input_size'][1:]
        )

        size_ood_train_data = len(ood_data)
        if size_ood_train_data < size_test_data:
            logger.info(
                f"There is still not sufficient OOD data in the training set"
                f" {size_ood_train_data}. Therefore, the size of the test set is going to decrease "
                f"for {ood_dataset} from {size_test_data} to {size_ood_train_data}")
            number_of_test_samples_decreased = True
            # backup_preds_test = np.copy(preds_test)
            # backup_logits_test = np.copy(logits_test)
            # backup_spk_count_test = np.copy(spk_count_test)

            backup_preds_test = preds_test.copy()
            backup_logits_test = logits_test.copy()
            backup_spk_count_test = spk_count_test.copy()

            preds_test = preds_test[:size_ood_train_data]
            logits_test = logits_test[:size_ood_train_data]
            spk_count_test = spk_count_test[:size_ood_train_data]

            # Define the new size for the test data for this OOD dataset
            size_test_data = len(logits_test)

        # Create the subset of the train OOD data, where it will have the same size as
        # the size of the test data.
        ood_loader = create_subset_of_specific_size_with_random_data(
            data=ood_data, size_data=size_ood_train_data, new_size=size_test_data,
            generator=g_ood, batch_size=batch_size_ood
        )

    else:  # size_ood_data > size_test_data
        logger.info(f"Reducing the number of samples for OOD dataset {ood_dataset} to match "
                    f"the number of samples of test data, equal to {size_test_data}")
        ood_loader = create_subset_of_specific_size_with_random_data(
            data=ood_data, size_data=size_ood_data, new_size=size_test_data,
            generator=g_ood, batch_size=batch_size_ood
        )


def fn_vs_bad(preds_test, test_labels, test_accuracy, distances_train_per_class, distances_test_per_class,
              class_names, df_columns, in_dataset, ood_dataset):
    # Reorder preds and test labels to match the order of in_or_out_distribution_per_tpr_test
    test_labels_per_predicted_class = []
    for class_index in range(10):
        test_labels_per_predicted_class.append(test_labels[find_idx_of_class(class_index, preds_test)])
    test_labels_reordered = np.concatenate(test_labels_per_predicted_class)
    preds_test_per_predicted_class = []
    for class_index in range(10):
        preds_test_per_predicted_class.append(preds_test[find_idx_of_class(class_index, preds_test)])
    preds_test_reordered = np.concatenate(preds_test_per_predicted_class)
    # Compare predictions and labels and output 1 where is correctly predicted, 0 where not
    correct_incorrect_clasification = np.where(preds_test_reordered == test_labels_reordered, 1, 0)

    # Obtain array with ind or ood decision for test instances and for specific TPR values
    # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
    dist_thresholds = thresholds_per_class_for_each_TPR(
        len(class_names), distances_train_per_class,
    )
    # Compute if test instances are classified as InD or OoD for every tpr
    in_or_out_distribution_per_tpr_test = compare_distances_per_class_to_distance_thr_per_class(
        distances_test_per_class,
        dist_thresholds
    )
    # Extract the list with only the TPR values we are interested in: 25, 50, 75 and 95 per cent
    tprs_to_extract = (25, 50, 75, 95)
    in_or_out_distribution_per_tpr_test = in_or_out_distribution_per_tpr_test[tprs_to_extract, :]

    # Now compare the test labels with the InD or OoD decision and obtain a [4, number_of_samples]
    # list, where 1 will mean the False Negative was correctly classified and 0 will mean
    # the False Negative was misclassified
    fn_correct_vs_incorrect_per_tpr = []
    for idx, in_or_out_one_tpr in enumerate(in_or_out_distribution_per_tpr_test):
        fn_position = np.where(in_or_out_one_tpr == 0)[0]
        fn_correct_vs_incorrect_per_tpr.append(
            np.where(correct_incorrect_clasification[fn_position] == 1, 1, 0)
        )

    df_fn_incorrect_vs_correct_one_dataset = pd.DataFrame(columns=df_columns)
    for i, fn_correct_vs_incorrect in enumerate(fn_correct_vs_incorrect_per_tpr):
        df_fn_incorrect_vs_correct_one_dataset.loc[len(df_fn_incorrect_vs_correct_one_dataset)] = [
            in_dataset,
            ood_dataset,
            len(preds_test),
            tprs_to_extract[i],
            len(fn_correct_vs_incorrect) / len(preds_test),
            len(fn_correct_vs_incorrect),
            len(np.nonzero(fn_correct_vs_incorrect)[0]) / len(fn_correct_vs_incorrect),
            (len(fn_correct_vs_incorrect) - len(np.nonzero(fn_correct_vs_incorrect)[0])) / len(fn_correct_vs_incorrect),
            test_accuracy,
        ]

    df_fn_vs_bad_classification = pd.concat(
        [df_fn_vs_bad_classification, df_fn_incorrect_vs_correct_one_dataset]
    )
    return df_fn_vs_bad_classification


def len_of_list_per_class(list_obj) -> int:
    s = 0
    for element in list_obj:
        s += len(element)
    return s
