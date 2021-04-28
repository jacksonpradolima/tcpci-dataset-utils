import os
import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
from tabulate import tabulate

from tcpci_dataset_utils.project_status import ProjectStatus
from pathlib import Path


class RunningProjectStatus(TestCase):
    def test_project_status(self):
        project_dir = f"../../data"
        dataset = "deeplearning4j@deeplearning4j"
        project_stat = ProjectStatus(project_dir, dataset)

        df = project_stat.get_summary()
        df_simple = df[["Name", "Period", "Builds", "Faults", "Tests", "Duration", "Interval"]]

        print(tabulate(df_simple, headers='keys', tablefmt='psql', showindex=False))

        self.assertTrue(True)

    @unittest.skip
    def test_plot_plots(self):
        project_dir = f"../../data"
        dataset = "deeplearning4j@deeplearning4j"
        project_stat = ProjectStatus(project_dir, dataset)

        outpath =  f"../../dataset_info{os.sep}{dataset}"
        Path(outpath).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(30, 20))
        project_stat.visualize_dataset_heatmap(fig, ax)
        plt.savefig(os.path.join(outpath, "heatpmap.pdf"), bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(15, 10))
        project_stat.visualize_dataset_info(ax)
        plt.savefig(os.path.join(outpath, "failures_by_cycle.pdf"), bbox_inches='tight')
        plt.clf()

        fig, ax = plt.subplots(figsize=(15, 10))
        project_stat.visualize_testcase_volatility(ax)
        plt.savefig(os.path.join(outpath, "testcase_volatility.pdf"), bbox_inches='tight')
        plt.clf()

        self.assertTrue(True)

    def test_save_project_status(self):
        project_dir = f"../../data"
        dataset = "deeplearning4j@deeplearning4j"
        project_stat = ProjectStatus(project_dir, dataset)

        outpath = f"../../dataset_info"
        Path(outpath).mkdir(parents=True, exist_ok=True)


        df = project_stat.get_summary()
        df_simple = df[["Name", "Period", "Builds", "Faults", "Tests", "Duration", "Interval"]]

        print(f"\n\nExporting Project Status to {outpath}{os.sep}project_status.txt")
        with open(f'{outpath}{os.sep}project_status.txt', 'w') as tf:
            tf.write(tabulate(df_simple, headers='keys', tablefmt='psql', showindex=False))

        print(f"Exporting Project Status to {outpath}{os.sep}project_status_table.tex")
        df_simple.to_latex(f'{outpath}{os.sep}project_status_table.tex', index=False)

        latex = df.to_latex(index=False)

        # Remove special characters provided by pandas
        latex = latex.replace("\\textbackslash ", "\\").replace(
            "\$", "$").replace("\{", "{").replace("\}", "}")

        # split lines into a list
        latex_list = latex.splitlines()
        caption = f"System Information"

        # Insert new LaTeX commands
        latex_list.insert(0, '\\begin{table*}[!ht]')
        latex_list.insert(1, f'\\caption{{{caption}}}')
        latex_list.insert(2, '\\resizebox{\\linewidth}{!}{')
        latex_list.append('}')
        latex_list.append('\\end{table*}')

        # join split lines to get the modified latex output string
        latex_new = '\n'.join(latex_list)

        # Save in a file
        print(f"Exporting Project Status to {outpath}{os.sep}project_status_table_complete.tex")
        with open(f'{outpath}{os.sep}project_status_table_complete.tex', 'w') as tf:
            tf.write(latex_new)

        self.assertTrue(True)
