import argparse
import csv
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd

from dateutil import parser
from utils.pandas_utils import split_dataframe_rows

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/data_filtering.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
)


def pre_processing(repository, force=False):
    """
    This function reads the travis json data and parse it a Filtered CSV with valid values
    (when the travis status is not 'errored', the log status is not 'unknown', and the tests ran)
    :param repository: The repository that will be analyzed
    """
    try:
        logging.debug(f"Start the filtering in '{repository}'")

        if os.path.isfile(f"{repository}/data-filtered.csv") and not force:
            logging.debug(f"File already exists! Skipping...")
            return

        df_repo = pd.read_json(f"{repository}/repo-data-travis.json")

        # convert columns to datetime
        df_repo.started_at = pd.to_datetime(df_repo.started_at)
        df_repo.finished_at = pd.to_datetime(df_repo.finished_at)

        # If the build have only the finished_at value
        df_repo.started_at.fillna(df_repo.finished_at, inplace=True)

        # Here we don't need the job_id
        df_repo = df_repo[['build_id', 'commit', 'status', 'started_at']]
        df_repo.rename(columns={'status': 'travis_status',
                                'started_at': 'travis_started_at'}, inplace=True)

        df_buildlog = pd.read_json(f"{repository}/buildlog-data-travis.json")
        df_buildlog.rename(columns={
            'tr_build_id': 'build_id',
            'tr_job_id': 'job_id',
            'tr_original_commit': 'commit',
            'tr_log_tests': 'tc_name',
            'tr_log_tests_num': 'tc_run',
            'tr_log_tests_duration': 'tc_duration',
            'tr_log_status': 'log_status',
            'tr_log_bool_tests_ran': 'bool_tests_ran'

        }, inplace=True)

        # create a DF with all test cases
        df = split_dataframe_rows(
            df_buildlog, ["tc_name", "tc_run", "tc_duration"], "#")

        logging.debug("All tests with were splitted")

        # create a DF with the errors
        df_error = split_dataframe_rows(df_buildlog[["build_id", "job_id", "commit",
                                                     "tr_log_tests_failed", "tr_log_tests_failed_num"]],
                                        ["tr_log_tests_failed", "tr_log_tests_failed_num"], "#")
        df_error.rename(columns={
            'tr_log_tests_failed': 'tc_name',
            'tr_log_tests_failed_num': 'tc_failed',

        }, inplace=True)

        logging.debug("All tests with errors were splitted")

        # The key columns from data
        # We can have a build with many jobs, so we want in the future the tc that failed max in any job of my build
        columns_id = ['build_id', 'commit', 'job_id', 'tc_name']

        # Remove the duplicates considering the key columns
        df_new = df[columns_id].drop_duplicates(subset=columns_id, keep="last")

        # Merge with Travis CI Information
        df_new = pd.merge(df_new, df_repo, on=[
                          'build_id', 'commit'], how='left')

        # Get the errors
        df_error = df_error.sort_values('tc_failed', ascending=False).drop_duplicates(
            subset=columns_id).sort_index()

        df_new = pd.merge(df_new, df_error, on=columns_id, how='left')

        # Get the next columns that we want to merge
        df = df[["build_id", "commit", 'job_id', 'tc_name',
                 'tc_run', 'tc_duration',
                 'log_status', 'bool_tests_ran']]

        # Columns with the build log status
        df_new = pd.merge(df_new,
                          df[['build_id', 'commit', 'job_id', 'tc_name',
                              'log_status', 'bool_tests_ran']],
                          on=columns_id, how='left')

        # Extract the following columns that have the largest value in case of duplicate
        df_temp = df[["build_id", "commit", 'job_id', 'tc_name', 'tc_run']] \
            .sort_values('tc_run', ascending=False).drop_duplicates(subset=columns_id).sort_index()
        df_new = pd.merge(df_new, df_temp, on=columns_id, how='left')

        df_temp = df[["build_id", "commit", 'job_id', 'tc_name', 'tc_duration']] \
            .sort_values('tc_duration', ascending=False).drop_duplicates(subset=columns_id).sort_index()
        df_new = pd.merge(df_new, df_temp, on=columns_id, how='left')

        df_new.loc[df_new["tc_run"].isnull(), 'tc_run'] = "0"
        df_new.loc[df_new["tc_duration"].isnull(), 'tc_duration'] = "0"
        df_new.loc[df_new["tc_failed"].isnull(), 'tc_failed'] = "0"

        # Get only valid data
        df_new = df_new.query(
            "travis_status != 'errored' and travis_status != 'canceled' and log_status != 'unknown' and bool_tests_ran == 1")

        # Copy to a new column
        df_new['build'] = df_new.build_id
        df_new = df_new.sort_values(by='travis_started_at')

        # Convert the test case names to a ID
        df_new['tc_id'] = pd.factorize(df_new.tc_name)[0] + 1
        # Convert the build_id to a new ID
        df_new['build_id'] = pd.factorize(df_new.build)[0] + 1

        # Select the columns
        df_new = df_new[['build_id', 'build', 'commit', 'travis_started_at',
                         'log_status', 'tc_id', 'tc_name', 'tc_duration', 'tc_run', 'tc_failed']]

        # Save the Filtered CSV
        df_new.to_csv(f'{repository}/data-filtered.csv', sep=";", index=False)
        logging.debug(f"Finish filtering, saving Filtered CSV for '{repository}'")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        logging.error(f'Failed to filtering CSV in line {exc_tb.tb_lineno}: {str(e)}')


def feature_engineering(repository, force=False):
    """
    This function get the Filtered Data and parse it to a file (features engineered CSV)
    that it will be used in the analysis
    :param repository: The repository that will be analyzed
    """
    try:
        logging.debug(f"Start the feature engineering procedure in '{repository}'")

        if os.path.isfile(f"{repository}/features-engineered.csv") and not force:
            logging.debug(f"File already exists! Skipping...")
            return

        df = pd.read_csv(
            '{}/data-filtered.csv'.format(repository), sep=";", parse_dates=True)

        # Select the main columns
        df = df[['build_id', 'tc_id', 'travis_started_at',
                 'tc_duration', 'tc_run', 'tc_failed']]

        # workaround to avoid possible errors
        # fix datetime (when we have +00:00 with the date)
        df['travis_started_at'] = df['travis_started_at'].apply(
            lambda date: datetime.datetime.strptime(date.split('+')[0] if '+' in date else date, '%Y-%m-%d %H:%M:%S'))
        df.travis_started_at = pd.to_datetime(df.travis_started_at)
        df.tc_run = pd.to_numeric(df.tc_run, errors='coerce')
        df.tc_failed = pd.to_numeric(df.tc_failed, errors='coerce')
        df.tc_duration = pd.to_numeric(df.tc_duration, errors='coerce')

        df = df[df.tc_run.notnull() & df.tc_failed.notnull()
                & df.tc_duration.notnull()]

        reddf = df.groupby(['build_id', 'tc_id'], as_index=False).agg(
            {'travis_started_at': np.min, 'tc_run': np.sum, 'tc_failed': np.sum, 'tc_duration': np.sum})

        tc_fieldnames = ['Id', 'Name', 'BuildId', 'Duration', 'CalcPrio', 'LastRun',
                         'NumRan', 'NumErrors', 'Verdict', 'Cycle', 'LastResults']

        tcdf = pd.DataFrame(columns=tc_fieldnames, index=reddf.index)

        # Id | Unique numeric identifier of the test execution
        tcdf['Id'] = reddf.index + 1
        # Name | Unique numeric identifier of the test case
        tcdf['Name'] = reddf['tc_id']
        # BuildId | Value uniquely identifying the build.
        tcdf['BuildId'] = reddf['build_id']
        # Duration | Approximated runtime of the test case
        tcdf['Duration'] = reddf['tc_duration']
        # CalcPrio | Priority of the test case, calculated by the prioritization algorithm(output column, initially 0)
        tcdf['CalcPrio'] = 0
        # LastRun | Previous last execution of the test case as date - time - string(Format: `YYYY - MM - DD HH: ii`)
        tcdf['LastRun'] = reddf['travis_started_at']
        # NumRan | Number of test ran
        tcdf['NumRan'] = reddf['tc_run']
        # Errors | Number of errors revealed
        tcdf['NumErrors'] = reddf['tc_failed']
        # Verdict | Test verdict of this test execution(Failed: 1, Passed: 0)
        tcdf['Verdict'] = reddf['tc_failed'].apply(lambda x: 1 if x > 0 else 0)

        # Let's order and process the data remain
        tcdf = tcdf.sort_values(by='LastRun')

        """
        This part of code can be used to provide compatibility with RETECS. 
        In your case, we want to analyze each build in Travis CI, in other others, each commit (inter-commits).
        We can "select" the correct column Build/Cycle inside the code.
        """
        # Cycle | The number of the CI cycle this test execution belongs to
        tcdf['monthdayhour'] = tcdf['LastRun'].apply(
            lambda x: (x.month, x.day, x.hour))
        tcdf['Cycle'] = pd.factorize(tcdf.monthdayhour)[0] + 1
        del tcdf['monthdayhour']

        logging.debug('... done')

        logging.debug(f"Store results in '{repository}/features-engineered.csv'")
        tcdf.to_csv(f"{repository}/features-engineered.csv", sep=';', na_rep='[]',
                    columns=tc_fieldnames,
                    header=True, index=False,
                    quoting=csv.QUOTE_NONE)

        logging.debug(f"Finish the feature engineering procedure in '{repository}'")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        logging.error(f'Failed to feature engineering in line {exc_tb.tb_lineno}: {str(e)}')


def add_last_results(repository, force=False):
    try:
        logging.debug(f"Start the feature engineering procedure in '{repository} to add last results'")

        if os.path.isfile(f"{repository}/features-engineered_lastresults.csv") and not force:
            logging.debug(f"File already exists! Skipping...")
            return

        df = pd.read_csv(
            '{}/features-engineered.csv'.format(repository), sep=";", parse_dates=True)

        logging.debug(
            'Collect historical test results (this takes some time)...')

        no_tcs = len(df.Name.unique())
        for tccount, name in enumerate(df.Name.unique(), start=1):
            verdicts = df.loc[df['Name'] == name, 'Verdict'].tolist()

            if len(verdicts) > 1:
                # LastResults | List of previous test results(Failed: 1, Passed: 0), ordered by ascending age.
                # Lists are delimited by [].
                df.loc[df['Name'] == name, 'LastResults'] = [None] + [verdicts[i::-1] for i in
                                                                      range(0, len(verdicts) - 1)]

            sys.stdout.write('\r%.2f%%' % (tccount / no_tcs * 100))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        logging.debug('... done')

        logging.debug(f"Store results in '{repository}/features-engineered_lastresults.csv'")
        df.to_csv(f"{repository}/features-engineered_lastresults.csv", sep=';', na_rep='[]',
                  header=True, index=False,
                  quoting=csv.QUOTE_NONE)

        logging.debug(f"Finish the feature engineering procedure in '{repository}' to add last results")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        logging.error(f'Failed to add last results in line {exc_tb.tb_lineno}: {str(e)}')


def parse_industrial_dataset(repository, force=False):
    try:
        logging.debug(f"Start the feature engineering procedure in '{repository}', an industrial dataset")

        if os.path.isfile(f"{repository}/features-engineered.csv") and not force:
            logging.debug(f"File already exists! Skipping...")
            return

        df = pd.read_csv(
            '{}/data-filtered.csv'.format(repository), sep=";", parse_dates=True)

        df.LastRun = pd.to_datetime(df.LastRun)

        cycles = pd.to_datetime(df['LastRun'].unique())

        df['BuildId'] = df['LastRun'].apply(
            lambda x: np.where(cycles == x)[0][0] + 1)

        # Drop duplicates and preserve rows that have the largest value in case of duplicate
        df = df.sort_values('Duration', ascending=False).drop_duplicates(subset=['Name', 'BuildId'],
                                                                         keep="last")
        # Order correctly
        df = df.sort_values('LastRun').sort_index()

        # Restart the Id
        df['Id'] = df.index + 1

        logging.debug(f"Store results in '{repository}/features-engineered.csv'")
        df.to_csv(f"{repository}/features-engineered.csv", sep=';', na_rep='[]',
                  header=True, index=False,
                  quoting=csv.QUOTE_NONE)

        logging.debug(f"Finish the feature engineering procedure in '{repository}', an industrial dataset")

    except Exception as e:
        logging.error(f'Failed to feature engineering (industrial dataset) in line {exc_tb.tb_lineno}: {str(e)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Extract commit details')

    ap.add_argument('-r', '--repository', dest='repository', type=str, required=True,
                    help='Complete directory of the project to analyse')

    args = ap.parse_args()
    pre_processing(args.repository)
    feature_engineering(args.repository)
    add_last_results(args.repository)