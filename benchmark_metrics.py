import argparse
from collections import OrderedDict
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from SCP.utils.common import load_config
from SCP.utils.plots import bayesian_test, cd_graph


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOD detection on SNNs", add_help=True)

    parser.add_argument("-c", "--conf", default="config", type=str, help="name of the configuration in config folder")
    parser.add_argument("-r", "--rope", default=0.01, type=float, help="ROPE: Region Of Practical Equivalence")
    parser.add_argument("-m", "--mode", default="one-file", type=str, choices=["one-file", "all-subdirectories"],
                        help="which mode to run")
    parser.add_argument("-f", "--file", type=str,
                        help="path of the file. If starts with '/' will be considered an absolute path")
    parser.add_argument("-d", "--dir", type=str,
                        help="Path of the directory to start the search from. Required if mode 'all-subdirectories'. "
                             "If starts with '/' will be considered an absolute path")
    parser.add_argument("-cd", "--cd-graph", action='store_true', dest='cd_graph',
                        help="if passed, CD graph is plotted")
    parser.add_argument("-bsr", "--bayesian-signrank", action='store_true', dest='bayesian_signrank',
                        help="if passed, bayesian signrank test is plotted")
    parser.add_argument("-bs", "--bayesian-signedtest", action='store_true', dest='bayesian_signed',
                        help="if passed, bayesian signed test is plotted")
    parser.add_argument("--metric", type=str, default='AUROC', help="Metric to use for ranking the methods",
                        choices=['AUROC', 'AUPR', 'FPR95'])
    parser.add_argument("--ref-ood-method", type=str, default='SCP', dest='ref_ood_method',
                        help="The method to make the pairwise bayesian tests",
                        choices=['Baseline', 'ODIN', 'Free energy', 'EnsembleOdinSCP', 'EnsembleOdinEnergy'])
    parser.add_argument("--ood-methods", default=['Baseline', 'ODIN', 'Free energy'], type=str, nargs='+',
                        dest="ood_methods", help="Test the specified methods against SCP. If empty, test all. "
                                                 "Options: Baseline, ODIN, Free energy, EnsembleOdinSCP, EnsembleOdinEnergy")
    parser.add_argument("--models", default=[], type=str, nargs='+', dest="models",
                        help="Test the specified models. If empty, test all. Options: 'Fully_connected' and 'ConvNet'")
    return parser


def check_tested_datasets_are_available(in_or_out, datasets_to_test, available_datasets, file_path: Path, indent):
    for dataset in datasets_to_test:
        if dataset not in available_datasets:
            print(f'{indent}The {in_or_out}distribution dataset {dataset} is not present in {file_path.name}')


def drop_available_datasets_that_wont_be_tested(in_or_out, df, datasets_to_test,
                                                available_datasets, indent) -> pd.DataFrame:
    for dataset in available_datasets:
        if dataset not in datasets_to_test:
            df = df[df['In-Distribution'] != dataset]
            print(f'{indent}The {in_or_out}distribution dataset {dataset} will be dropped'
                  f' as is not present in the config file')
    return df


def check_tested_datasets_are_available_and_drop_not_tested_datasets(config, df, file_path, indent) -> pd.DataFrame:
    in_dataset_to_test = config["in_distribution_datasets"]
    ood_dataset_to_test = config["out_of_distribution_datasets"]

    available_in_datasets = df['In-Distribution']
    available_ood_datasets = df['Out-Distribution']

    check_tested_datasets_are_available('in-', in_dataset_to_test, available_in_datasets, file_path, indent)
    check_tested_datasets_are_available('out-of-', ood_dataset_to_test, available_ood_datasets, file_path, indent)

    df = drop_available_datasets_that_wont_be_tested('in-', df, in_dataset_to_test, available_in_datasets, indent)
    df = drop_available_datasets_that_wont_be_tested('out-of-', df, ood_dataset_to_test, available_ood_datasets, indent)

    return df


def save_plots_one_file(file_path: Path, config: dict, models: List, args: argparse.Namespace,
                        save_figure_path: Path, indent=''):
    # Load results form the file
    df_full_results = pd.read_excel(file_path)

    if not all(
            [
                item in list(
                    df_full_results.columns
                ) for item in ['In-Distribution', 'Out-Distribution', 'OoD Method', 'Model']
            ]
    ):
        print(f'{indent}The file {file_path} will be ignored as is not a valid results .xlsx file')
        return

    df_full_results = check_tested_datasets_are_available_and_drop_not_tested_datasets(
        config, df_full_results, file_path, indent
    )

    if len(df_full_results) == 0:
        print(f'{indent}* ----------------------------------------------------------- *')
        print(f'{indent}Exiting {file_path.name} as no tested datasets have been found')
        print(f'{indent}* ----------------------------------------------------------- *')
        return

    # As 'Ours' is the name of our method in results but we want to use SCP for the plots, we need to
    # create this variable containing 'Ours' when SCP is the reference method. We also need to remove from
    # ood_methods the reference method
    ref_method_df_name = args.ref_ood_method if args.ref_ood_method != 'SCP' else 'Ours'
    if args.ref_ood_method in args.ood_methods:
        print(f'{indent}Removing from the results the cases where the '
              f'In-Distribution and Out-of-Distribution dataset match')
        args.ood_methods.remove(args.ref_ood_method)

    # Initialize the dict containing the scores per ood_method
    scores_dict = OrderedDict()

    df = df_full_results.loc[df_full_results['In-Distribution'] != df_full_results['Out-Distribution']]

    models_available = df_full_results['Model'].unique()
    for model in models:

        if model not in models_available:
            continue

        df_ref_method = df.loc[df['OoD Method'] == ref_method_df_name]
        scores_reference_method = df_ref_method.loc[df_ref_method['Model'] == model][[args.metric]].to_numpy() / 100

        scores_dict[args.ref_ood_method] = scores_reference_method.squeeze()

        for method in args.ood_methods:
            df_other_method = df.loc[df['OoD Method'] == method]
            score_other_method = df_other_method.loc[df_other_method['Model'] == model][[args.metric]].to_numpy() / 100
            scores_dict[method] = score_other_method.squeeze()

        if args.cd_graph:
            cd_graph(scores_dict, fig_path=save_figure_path)

        if args.bayesian_signrank or args.bayesian_signed:

            for method in args.ood_methods:
                pairwise_scores = OrderedDict()
                pairwise_scores[args.ref_ood_method] = scores_dict[args.ref_ood_method]
                pairwise_scores[method] = scores_dict[method]

                if save_figure_path.is_dir():
                    figure_path = save_figure_path / f'{args.ref_ood_method}vs{method}'
                else:
                    figure_path = Path(f'{save_figure_path.as_posix()}_{args.ref_ood_method}vs{method}')

                if args.bayesian_signrank:
                    bayesian_test(pairwise_scores, option='signrank', fig_path=figure_path, rope=args.rope)
                if args.bayesian_signed:
                    bayesian_test(pairwise_scores, option='signtest', fig_path=figure_path, rope=args.rope)


def all_subdirectories():
    """
    Recorrer todos los subdirectorios de una carpeta y crear en la carpeta donde haya un excel que empiece por
    benchmark_ y en la carpeta indicada a traves de los args los test indicados (cd, bayesian) de los datasets
    indicados y los metodos indicados.
    """
    pass


def open_folders_and_plot_excels(parent_folder: Path, config: dict, models: List, args: argparse.Namespace, indent=''):
    for element in parent_folder.iterdir():
        if element.is_dir():
            print(f'{indent}Opening {element.name} folder')
            indent += '\t'
            open_folders_and_plot_excels(element, config, models, args, indent=indent)
            indent = indent[:-1]
        else:
            if element.suffix == ".xlsx":
                print(f'{indent}Processing {element.name}')
                save_plots_one_file(
                    element, config, models, args, save_figure_path=element.with_stem(element.stem), indent=indent
                )


def main(args: argparse.Namespace):
    print(f'Loading configuration from {args.conf}.toml')
    config = load_config(args.conf)

    print(f'The following In-Distribution datasets will be tested: {config["in_distribution_datasets"]}')
    print(f'The following Out-of-Distribution datasets will be tested: {config["out_of_distribution_datasets"]}')

    if not args.models:
        models = ['Fully_connected', 'ConvNet']
    else:
        models = args.models

    if args.mode == "one-file":
        file_path = Path(args.file)
        save_plots_one_file(file_path, config, models, args, save_figure_path=file_path.parent)

    elif args.mode == "all-subdirectories":
        assert dir, '--dir argument must be specified when using "all-subdirectories" option'
        parent_folder_path = Path(args.dir)
        print(f'Starting the search of all results files from {parent_folder_path}')
        open_folders_and_plot_excels(parent_folder_path, config, models, args)

    else:
        NameError('Wrong option')


if __name__ == "__main__":
    # args =
    main(get_args_parser().parse_args())