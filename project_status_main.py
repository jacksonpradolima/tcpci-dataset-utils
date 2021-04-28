import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from pathlib import Path

from alive_progress import alive_bar

from tcpci_dataset_utils.project_status import ProjectStatus

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

summary_cols = ["Name", "Period", "Builds",
                "Faults", "FaultsByCycle",
                "Tests", "Volatility",
                "Duration", "Interval"]

DEFAULT_EXPERIMENT_DIR = 'dataset_info'

def plot_project_status(project_stat: ProjectStatus, path):
    Path(path).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(30, 20))
    project_stat.visualize_dataset_heatmap(fig, ax)
    plt.savefig(os.path.join(path, "heatpmap.pdf"), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 10))
    project_stat.visualize_dataset_info(ax)
    plt.savefig(os.path.join(path, "failures_by_cycle.pdf"),
                bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 10))
    project_stat.visualize_testcase_volatility(ax)
    plt.savefig(os.path.join(path, "testcase_volatility.pdf"),
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Project Status - Observing the Project')

    ap.add_argument('--project_dir', required=True)
    ap.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')

    ap.add_argument('-o', '--output_dir', default=DEFAULT_EXPERIMENT_DIR)

    args = ap.parse_args()

    # Create a new dataframe to save the project status
    df = pd.DataFrame(columns=summary_cols)

    with alive_bar(len(args.datasets), title="Project Status") as bar:
        # Get the status for each variant from project
        bar.text('Processing the project status for each dataset...')
        bar()
        for dataset in args.datasets:
            outpath = f"{args.output_dir}{os.sep}{dataset}"
            project_stat = ProjectStatus(args.project_dir, dataset)

            df = df.append(project_stat.get_summary())
            plot_project_status(project_stat, outpath)

            bar()

    df.sort_values(by=['Name'], inplace=True)

    df_simple = df[["Name", "Period", "Builds", "Faults", "Tests", "Duration", "Interval"]]

    print(f"\n\nExporting Project Status to {args.output_dir}{os.sep}project_status.txt")
    with open(f'{args.output_dir}{os.sep}project_status.txt', 'w') as tf:
        tf.write(tabulate(df_simple, headers='keys', tablefmt='psql', showindex=False))

    print(f"Exporting Project Status to {args.output_dir}{os.sep}project_status_table.tex")
    df_simple.to_latex(f'{args.output_dir}{os.sep}project_status_table.tex', index=False)

    latex = df.to_latex(index=False)

    # Remove special characters provided by pandas
    latex = latex.replace("\\textbackslash ", "\\").replace(
        "\$", "$").replace("\{", "{").replace("\}", "}")

    # split lines into a list
    latex_list = latex.splitlines()
    caption = f"System Information"

    # Insert new LaTeX commands
    latex_list.insert(0, '\\begin{table}[!ht]')
    latex_list.insert(1, '\\addtolength{\\tabcolsep}{-4pt}')
    # latex_list.insert(1, f'\\caption{{{caption}}}')
    # latex_list.insert(2, '\\resizebox{\\linewidth}{!}{')
    # latex_list.append('}')
    latex_list.append('\\end{table}')

    # join split lines to get the modified latex output string
    latex_new = '\n'.join(latex_list)

    # Save in a file
    print(f"Exporting Project Status to {args.output_dir}{os.sep}project_status_table_complete.tex")
    with open(f'{args.output_dir}{os.sep}project_status_table_complete.tex', 'w') as tf:
        tf.write(latex_new)