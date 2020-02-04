import os
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

# set ggplot style
plt.style.use('ggplot')
mpl.rcParams.update({'font.size': 24})

sns.set_style("whitegrid")

USE_LATEX = False
DEFAULT_EXPERIMENT_DIR = 'results/dataset_info/'

repos = {
    'alibaba@druid': 'Druid',
    'alibaba@fastjson': 'Fastjson',
    'deeplearning4j@deeplearning4j': 'Deeplearning4j',
    'DSpace@DSpace': 'DSpace',
    'HubSpot@Singularity': 'Singularity',
    'google@guava': 'Guava',
    'iluwatar@java-design-patterns': 'Java Design Patterns',
    'spring-projects@spring-boot': 'Spring Boot',
    'square@okhttp': 'OkHttp',
    'square@retrofit': 'Retrofit',
    'zxing@zxing': 'ZXing',
    'paintcontrol': 'Paint Control',
    'sakaiproject@sakai': 'Sakai',
    'iofrol': 'IOF/ROL',
    'gsdtsr': 'GSDTSR',
    'lexisnexis': 'LexisNexis'
}

industrial_datasets = ['iofrol', 'paintcontrol', 'gsdtsr', 'lexisnexis']

if os.environ.get('DISPLAY', '') == '':
    # echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


def save_figures(filename, output_dir):
    plt.savefig(os.path.join(output_dir, filename +
                             ('.pgf' if USE_LATEX else '.pdf')), bbox_inches='tight')


def visualize_dataset_info(dataset, dataset_dir, output_dir):
    filename = f"failures_by_cycle"

    faults = 'NumErrors'

    if dataset in industrial_datasets:
        faults = 'Verdict'

    if not os.path.exists(os.path.join(output_dir, filename + ('.pgf' if USE_LATEX else '.pdf'))):
        df = pd.read_csv(f'{dataset_dir}/{dataset}/features-engineered.csv', sep=';', thousands=',')
        df_group = df.groupby(['BuildId'], as_index=False).agg(
            {faults: np.sum, 'Duration': np.mean})
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_xlabel('CI Cycle')
        ax.set_ylabel('')
        df_group[[faults]].plot(ax=ax)
        plt.legend([f"Total Failures: {int(df_group[faults].sum())}"], loc='center left', bbox_to_anchor=(0.655, -0.10))
        plt.tight_layout()
        save_figures(filename, output_dir)
        plt.clf()


def visualize_testcase_volatility(dataset, dataset_dir, output_dir):
    filename = f"testcase_volatility"

    df = pd.read_csv(f'{dataset_dir}/{dataset}/features-engineered.csv', sep=';', thousands=',')
    df_group = df.groupby(['BuildId'], as_index=False).agg({'Name': 'count'})

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlabel('CI Cycle')
    ax.set_ylabel('Number of Test Cases')
    df_group[["Name"]].plot(ax=ax)
    ax.legend().set_visible(False)
    plt.tight_layout()
    save_figures(filename, output_dir)
    plt.clf()


def visualize_dataset_heatmap(dataset, dataset_dir, output_dir, ignore=[], attribute="NumErrors"):
    if dataset in ignore:
        return

    filename = f"heatmap_{attribute}"

    if attribute == "NumErrors" and dataset in industrial_datasets:
        attribute = 'Verdict'

    if not os.path.exists(os.path.join(output_dir, filename + ('.pgf' if USE_LATEX else '.pdf'))):
        df = pd.read_csv(f'{dataset_dir}/{dataset}/features-engineered.csv', sep=';', thousands=',')

        bid = df.BuildId.values
        nerr = df[attribute].values.astype(np.int)
        name = df.Name.values

        # The Test Case ID showed in the plot are not the real Test Case Name
        # Here a new id is created
        # Note that there is no warranty of the order of Test Case Number. Also,
        # the names not necessarily are numbers
        tc2idx = {}
        for n in name:
            if not n in tc2idx:
                tc2idx[n] = len(tc2idx)
        total_use_cases = len(tc2idx)

        # Not executed tests are noted as -1 value
        data = np.zeros((len(set(bid)), total_use_cases))-1
        for i, b in enumerate(bid):
            n = tc2idx[name[i]]
            e = nerr[i]
            data[b-1, n] = e

        data = data.astype(np.int)

        set_errors = sorted(list(np.unique(data[data >= 0])))
        ticks_names = ['Not exist in T'] + ["{}".format(x)
                                            for x in set_errors]

        ticks_values = [-1] + [x for x in set_errors]

        offset = len(set_errors)/3

        # Ignore the "gaps" in gradient
        # Map the original heatmapvalues to a discrete domain
        set_ticks = sorted(list(np.unique(data)))

        data_copy = data.copy()
        for t in set_ticks:
            updt_val = set_ticks.index(t)
            if t < 0:
                updt_val += offset*t
            if t > 0:
                updt_val += offset

            data_copy[data == t] = updt_val

        data = data_copy
        ticks_values = sorted(list(np.unique(data)))
        fig, ax = plt.subplots(figsize=(30, 20))
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='5%', pad=0.05)

        cmap = "binary"
        img = ax.imshow(data.T, cmap=cmap, interpolation=None)

        clb = fig.colorbar(img, cax=cbar_ax)
        max_ticks = 13  # Limit number of ticks in colorbar
        idx_to_show = [0]  # Show for sure the two firts ticks

        idx_to_show += list(np.arange(1, len(ticks_values),
                                      int(len(ticks_values)/min(len(ticks_values),
                                                                max_ticks))))
        clb.set_ticks(np.array(ticks_values)[idx_to_show])
        clb.set_ticklabels(np.array(ticks_names)[idx_to_show])

        clb.outline.set_edgecolor('black')
        clb.outline.set_linewidth(1)

        ax.set_xlabel("CI Cycle")
        ax.set_ylabel("Test Case Id")
        ax.set_aspect("auto")

        ax.invert_yaxis()
        ax.grid(b=None)
        cbar_ax.grid(b=None)
        ax.locator_params(axis='y', nbins=5)
        ax.locator_params(axis='x', nbins=5)

        save_figures(filename, output_dir)


def generate_summary(datasets, dataset_dir, output_dir, plot=True):
    summary_cols = ["Name", "Period", "Builds", "Faults",
                    "Faulty builds", "Tests", "Test suite size"]
    summary = pd.DataFrame(columns=summary_cols)

    r1_cols = ['Name', 'Duration']
    r1 = pd.DataFrame(columns=r1_cols)

    r2_cols = ['Name', 'Interval']
    r2 = pd.DataFrame(columns=r2_cols)

    for dataset in datasets:
        print(f"Analysing '{dataset}'")

        faults, duration, diff_date, total_builds = 'NumErrors', [], [], 0

        if dataset in industrial_datasets:
            faults = 'Verdict'

            df = pd.read_csv(f'{dataset_dir}/{dataset}/features-engineered.csv', sep=';', thousands=',')

            df.LastRun = pd.to_datetime(df.LastRun)

            # Sort by commits arrival and start
            df = df.sort_values(by=['LastRun'])

            # Convert to minutes only valid build duration
            duration = [x / 60 for x in df.Duration.tolist()]

            dates = pd.to_datetime(df['LastRun'].unique())

            # Difference between commits arrival - Convert to minutes to improve the view
            diff_date = [(dates[i] - dates[i + 1]).seconds /
                         60 for i in range(len(dates) - 1)]

            total_builds = df.BuildId.nunique()
        else:
            print(f"{dataset_dir}/{dataset}/repo-data-travis.json")
            df_repo = pd.read_json(f"{dataset_dir}/{dataset}/repo-data-travis.json")

            # convert columns to datetime
            df_repo.started_at = pd.to_datetime(df_repo.started_at)
            df_repo.finished_at = pd.to_datetime(df_repo.finished_at)

            # If the build was canceled, we have only the finished_at value
            df_repo.started_at.fillna(df_repo.finished_at, inplace=True)

            # Sort by commits arrival and start
            df_repo = df_repo.sort_values(by=['started_at'])

            # Convert to minutes only valid build duration
            # TODO: Removed one "outlier" from Google Guava
            # duration = [x / 60 for x in df_repo.duration.tolist() if x > 0 and x / 60 < 1000]
            duration = [x / 60 for x in df_repo.duration.tolist()]

            # Difference between commits arrival - Convert to minutes to improve the view
            diff_date = [(df_repo.started_at[i] - df_repo.started_at[i + 1]).seconds / 60 for i in
                         range(len(df_repo.started_at) - 1)]

            total_builds = df_repo.build_id.nunique()

            df = pd.read_csv(f'{dataset_dir}/{dataset}/features-engineered.csv', sep=';', thousands=',')
            df.LastRun = pd.to_datetime(df.LastRun)

        # Buld Period
        mindate = df['LastRun'].min().strftime("%Y/%m/%d")
        maxdate = df['LastRun'].max().strftime("%Y/%m/%d")

        builds = df['BuildId'].max()

        faults = df.query('Verdict > 0').groupby(
            ['BuildId'], as_index=False).agg({faults: np.sum})[faults]

        # Number of builds in which at least one test failed
        faulty_builds = faults.count()

        # Total of faults
        total_faults = faults.sum()

        # number of unique tests identified from build logs
        test_max = df['Name'].nunique()

        tests = df.groupby(['BuildId'], as_index=False).agg({'Name': 'count'})

        # range of tests executed during builds
        test_suite_min = tests['Name'].min()
        test_suite_max = tests['Name'].max()

        row = [dataset, mindate + "-" + maxdate, f"{total_builds} ({builds})", total_faults, faulty_builds,
               test_max, str(test_suite_min) + "-" + str(test_suite_max)]

        summary = summary.append(pd.DataFrame(
            [row], columns=summary_cols), ignore_index=True)

        repo_name = repos[dataset]
        r1 = r1.append(pd.DataFrame(
            list([[repo_name, v] for v in duration]), columns=r1_cols), ignore_index=True)
        r2 = r2.append(pd.DataFrame(
            list([[repo_name, v] for v in diff_date]), columns=r2_cols), ignore_index=True)

    r1 = r1.sort_values(by='Name')
    r2 = r2.sort_values(by='Name')

    print("\n================= Duration =================")
    print_mean(r1, "Duration")

    print("\n================= Interval =================")
    print_mean(r2, "Interval")
    print("\n")

    _plot_repo_analysis(r1, 'Duration', "build_duration" if len(datasets) > 1 else f"build_duration_{datasets[0]}", output_dir)
    _plot_repo_analysis(r2, 'Interval', "diff_date" if len(datasets) > 1 else f"diff_date_{datasets[0]}", output_dir)

    return summary


def _plot_repo_analysis(data, y_col, filename, output_dir):
    filename = f"repo_{filename}_boxplot"

    if not os.path.exists(os.path.join(output_dir, filename + ('.pgf' if USE_LATEX else '.pdf'))):
        plt.figure(figsize=(30, 10))
        ax = sns.boxplot(x='Name', y=y_col, data=data)

        medians = data.groupby(['Name'])[y_col].median().values
        median_labels = [str(np.round(s, 2)) for s in medians]

        pos = range(len(medians))
        for tick, label in zip(pos, ax.get_xticklabels()):
            ax.text(pos[tick], medians[tick] + 0.5, median_labels[tick],
                    horizontalalignment='center', size='x-small', color='black', weight='semibold')

        plt.title("")
        plt.suptitle("")
        plt.xlabel('')
        plt.ylabel('Delay (min)')
        save_figures(filename, output_dir)
        plt.clf()


def print_mean(df, column):
    mean = df.groupby(['Name'], as_index=False).agg({column: ['mean', 'std']})
    mean.columns = ['Name', 'mean', 'std']

    # Round values (to be used in the article)
    mean = mean.round({'mean': 4, 'std': 3})
    mean = mean.infer_objects()

    print(tabulate(mean, headers='keys', tablefmt='psql', showindex=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Project Status')

    ap.add_argument('--is_visualization', default=False,
                    type=lambda x: (str(x).lower() == 'true'))

    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')

    ap.add_argument('-o', '--output_dir', default=DEFAULT_EXPERIMENT_DIR)

    args = ap.parse_args()

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.is_visualization:
        output_dir_temp = output_dir
        for dataset in args.datasets:
            print(f"Ploting for {dataset} dataset")
            output_dir = os.path.join(output_dir_temp, dataset)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            visualize_dataset_info(dataset, args.dataset_dir, output_dir)
            visualize_testcase_volatility(
                dataset, args.dataset_dir, output_dir)

            heatmap_ignored_datasets = [
                #"gsdtsr"
            ]

            attr_to_visualize = ["NumErrors", "Duration"]
            for attr in attr_to_visualize:
                visualize_dataset_heatmap(dataset, args.dataset_dir, output_dir,
                                          ignore=heatmap_ignored_datasets,
                                          attribute=attr)

    else:
        df = generate_summary(args.datasets, args.dataset_dir, args.output_dir)
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        print(df.to_latex())
