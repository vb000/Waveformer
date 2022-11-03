import os, sys
import argparse
import shutil

from tqdm import tqdm
import pandas as pd

def meta_csv_to_dict(meta):
    """
    Convert a ['fname', 'label'] headed dataframe to
    a dict with labels as keys, and list of file names as values.
    """
    samples_dict = {}
    samples = pd.read_csv(meta)
    labels = list(samples['label'].unique())
    for l in labels:
        samples_dict[l] = list(samples.loc[samples['label'] == l]['fname'])
    return samples_dict

def write_scaper_source(dset, data_dir, meta_dir, out_dir):
    src_dir = os.path.abspath(data_dir)
    file_list_csv = os.path.join(meta_dir, '%s.csv' % dset)
    samples_dict = meta_csv_to_dict(file_list_csv)
    print("Creating %s set..." % dset)
    for l in tqdm(samples_dict):
        dest_dir = os.path.join(out_dir, dset, l)
        assert not os.path.exists(dest_dir), \
            "Ouput dir %s already exists" % dest_dir
        os.makedirs(dest_dir)

        file_list = samples_dict[l]
        src_file_list = [os.path.join(src_dir, _) for _ in file_list]
        dest_file_list = [os.path.join(dest_dir, os.path.basename(_)) for _ in file_list]

        for s, d in zip(src_file_list, dest_file_list):
            shutil.copy(s, d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_dir', type=str, default=None,
        help="Path to FSDKaggle2018 dataset.")
    parser.add_argument(
        'meta_dir', type=str, default=None,
        help="Path to directory with csv file lists.")
    parser.add_argument(
        'output_dir', type=str, default=None,
        help="Path to the output directory."
    )
    args = parser.parse_args()

    for dset in ['train', 'val', 'test']:
        write_scaper_source(
            dset=dset, data_dir=args.data_dir, meta_dir=args.meta_dir,
            out_dir=args.output_dir)
