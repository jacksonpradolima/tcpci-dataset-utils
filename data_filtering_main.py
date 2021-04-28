import argparse
import logging
import os

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/data_filtering.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

from tcpci_dataset_utils.data_filtering import DataFiltering

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Extract commit details')

    ap.add_argument('-r', '--repository', dest='repository', type=str, required=True,
                    help='Complete directory of the project to analyse')

    args = ap.parse_args()

    filt = DataFiltering(args.repository)
    filt.pre_processing()
    filt.feature_engineering()
    filt.add_last_results()
