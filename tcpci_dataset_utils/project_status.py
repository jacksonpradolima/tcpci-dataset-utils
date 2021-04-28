import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler

# set ggplot style
plt.style.use('ggplot')
sns.set_style("whitegrid")

mpl.rcParams.update({'font.size': 24})

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class ProjectStatus(object):
    def __init__(self, project_dir, project):
        self.project_dir = project_dir
        self.project = project
        self.scaler = MinMaxScaler()

    def update_project(self, project_dir, project):
        self.project_dir = project_dir
        self.project = project

    def _faults_column_attribute(self, df):
        return 'NumErrors' if 'NumErrors' in df.columns else 'Verdict'

    def get_summary(self):
        def get_sparline(dataframe):
            # Sparklines
            sparklines = "\\sparkspike 0 0"  # Workaround for variants without failures
            if len(dataframe) > 1:
                scaled_values = self.scaler.fit_transform(dataframe)

                # sparklines = f'    '.join([f"\\sparkspike {i[0]} {i[1]}" for i in scaled_values])
                sparklines = f'    '.join([f"{i[0]} {i[1]}" for i in scaled_values])

                # return "\\begin{sparkline}{15} " + os.linesep + sparklines + os.linesep + " \\end{sparkline}"
                return "\\begin{sparkline}{15} " + "\\spark " + sparklines + " / \\end{sparkline}"

            return ""

        summary_cols = ["Name", "Period", "Builds",
                        "Faults", "FaultsByCycle",
                        "Tests", "Volatility",
                        "Duration", "Interval"]

        summary = pd.DataFrame(columns=summary_cols)

        df = pd.read_csv(f'{self.project_dir}{os.sep}{self.project}{os.sep}features-engineered.csv', sep=';', thousands=',')
        faults_col = self._faults_column_attribute(df)
        df['LastRun'] = pd.to_datetime(df['LastRun'])

        # Sort by commits arrival and start
        df = df.sort_values(by=['LastRun'])

        if 'NumErrors' in faults_col:
            # We have the entire information from json file.
            # Because the features-engineered.csv contains only information after a post processing (filtering)
            df_repo = pd.read_json(f"{self.project_dir}{os.sep}{self.project}{os.sep}repo-data-travis.json")

            # convert columns to datetime
            df_repo.started_at = pd.to_datetime(df_repo.started_at)
            df_repo.finished_at = pd.to_datetime(df_repo.finished_at)

            # If the build was canceled, we have only the finished_at value
            df_repo.started_at.fillna(df_repo.finished_at, inplace=True)

            # Sort by commits arrival and start
            df_repo = df_repo.sort_values(by=['started_at'])

            df_repo.duration = df_repo.duration.fillna(0)
            duration = [x / 60 for x in df_repo.duration.tolist()]

            # Difference between commits arrival - Convert to minutes to improve the view
            diff_date = [(df_repo.started_at[i] - df_repo.started_at[i + 1]).seconds / 60 for i in
                         range(len(df_repo.started_at) - 1)]

            total_builds = df_repo.build_id.nunique()
        else:
            # Convert to minutes only valid build duration
            duration = [x / 60 for x in df.Duration.tolist()]

            dates = pd.to_datetime(df['LastRun'].unique())

            # Difference between commits arrival - Convert to minutes to improve the view
            diff_date = [(dates[i] - dates[i + 1]).seconds /
                         60 for i in range(len(dates) - 1)]

            total_builds = df.BuildId.nunique()

        # Buld Period
        mindate = df['LastRun'].min().strftime("%Y/%m/%d")
        maxdate = df['LastRun'].max().strftime("%Y/%m/%d")

        faults = df.query('Verdict > 0').groupby(['BuildId'], as_index=False).agg({faults_col: np.sum})

        # Number of builds in which at least one test failed
        faulty_builds = faults[faults_col].count() if len(faults) > 0 else 0

        # Total of faults
        total_faults = faults[faults_col].sum() if len(faults) > 0 else 0

        # number of unique tests identified from build logs
        test_max = df['Name'].nunique()

        tests = df.groupby(['BuildId'], as_index=False).agg({'Name': 'count'})

        # range of tests executed during builds
        test_suite_min = tests['Name'].min()
        test_suite_max = tests['Name'].max()

        builds = df['BuildId'].max()

        # Sparklines
        sparklines_faults = get_sparline(faults[['BuildId', faults_col]]) if total_faults > 0 else ""
        sparklines_volatility = get_sparline(tests[['BuildId', 'Name']])

        row = [self.project.split("@")[-1].title(), mindate + "-" + maxdate,
               f"{total_builds} ({builds})",
               f"{total_faults} ({faulty_builds})",
               sparklines_faults,
               f"{test_max} ({test_suite_min} - {test_suite_max})",
               sparklines_volatility,
               f"{round(np.mean(duration), 2)} ({round(np.std(duration), 2)})" if len(duration) > 0 else "-",
               f"{round(np.mean(diff_date), 2)} ({round(np.std(diff_date), 2)})" if len(diff_date) > 0 else "-"]

        summary = summary.append(pd.DataFrame(
            [row], columns=summary_cols), ignore_index=True)

        return summary

    def visualize_dataset_heatmap(self, fig, ax):
        df = pd.read_csv(f'{self.project_dir}{os.sep}{self.project}{os.sep}features-engineered.csv', sep=';', thousands=',')
        faults_col = self._faults_column_attribute(df)

        bid = df.BuildId.values
        nerr = df[faults_col].values.astype(np.int)
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
        data = np.zeros((len(set(bid)), total_use_cases)) - 1
        for i, b in enumerate(bid):
            n = tc2idx[name[i]]
            e = nerr[i]
            data[b - 1, n] = e

        data = data.astype(np.int)

        set_errors = sorted(list(np.unique(data[data >= 0])))
        ticks_names = ['Not exist in T'] + ["{}".format(x)
                                            for x in set_errors]

        # Ignore the "gaps" in gradient
        # Map the original heatmapvalues to a discrete domain
        set_ticks = sorted(list(np.unique(data)))

        offset = len(set_errors) / 3

        data_copy = data.copy()
        for t in set_ticks:
            updt_val = set_ticks.index(t)
            if t < 0:
                updt_val += offset * t
            if t > 0:
                updt_val += offset

            data_copy[data == t] = updt_val

        data = data_copy
        ticks_values = sorted(list(np.unique(data)))

        ##########
        # Figure #
        ##########
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='5%', pad=0.05)

        img = ax.imshow(data.T, cmap="binary", interpolation=None)

        clb = fig.colorbar(img, cax=cbar_ax)
        max_ticks = 13  # Limit number of ticks in colorbar
        idx_to_show = [0]  # Show for sure the two firts ticks

        idx_to_show += list(np.arange(1, len(ticks_values),
                                      int(len(ticks_values) / min(len(ticks_values),
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

    def visualize_dataset_info(self, ax):
        df = pd.read_csv(f'{self.project_dir}{os.sep}{self.project}{os.sep}features-engineered.csv', sep=';', thousands=',')
        faults_col = self._faults_column_attribute(df)

        df_group = df.groupby(['BuildId'], as_index=False).agg({faults_col: np.sum, 'Duration': np.mean})



        ax.set_xlabel('CI Cycle')
        ax.set_ylabel('Number of Failures')
        df_group[[faults_col]].plot(ax=ax)
        plt.legend([f"Total Failures: {int(df_group[faults_col].sum())}"],
                   loc='center left',
                   bbox_to_anchor=(0.655, -0.10))
        plt.tight_layout()

    def visualize_testcase_volatility(self, ax):
        df = pd.read_csv(f'{self.project_dir}{os.sep}{self.project}{os.sep}features-engineered.csv', sep=';', thousands=',')
        df_group = df.groupby(['BuildId'], as_index=False).agg({'Name': 'count'})

        ax.set_xlabel('CI Cycle')
        ax.set_ylabel('Number of Test Cases')
        df_group[["Name"]].plot(ax=ax)
        ax.legend().set_visible(False)
        plt.tight_layout()
