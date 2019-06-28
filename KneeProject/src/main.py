from preprocessing import *
import sys
import argparse

parser = argparse.ArgumentParser(description='Month folder name such as 00m,12m,24m etc.')

parser.add_argument('-month','--month',help='Month folder name')


def main(month):
    read_dicome_and_process(content_file_path='/gpfs/data/denizlab/Datasets/OAI_original/', month=month, method='mean',
                            save_dir='/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_mean/')
if __name__ == '__main__':
    args = parser.parse_args()
    month = args.month
    main(month)
    print(month)