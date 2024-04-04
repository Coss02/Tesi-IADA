from copy import deepcopy
from avalanche.benchmarks.utils.utils import concat_datasets
from os.path import join
from utils.fm import my_load, my_save
import os
import json
import datetime
import numpy as np
from sklearn.feature_extraction import DictVectorizer

def count_sample_in_dataset(dataset):
    return len(dataset)
def count_sample_in_experience(experience):
    return count_sample_in_dataset(experience.dataset)

def count_sample_in_stream(stream):
    count_list = []
    for exp in stream:
        count_list.append(count_sample_in_experience(exp))
    return count_list

def get_full_dataset_experience(stream):
    full_exp = deepcopy(stream[0])
    full_exp.dataset = concat_datasets([exp.dataset for exp in stream])
    return full_exp


DS_ROOT_PATH = 'datasets/drebin_features'
def load_drebin_dataset(ds_root_path: str = DS_ROOT_PATH,
                 X_fname: str = 'extended-features-X-updated-reduced-1k.json',
                 y_fname: str = 'extended-features-y-updated.json',
                 meta_fname: str = 'extended-features-meta-updated.json',
                 num_features: str = '1k'):
    """
    The function to load features in the Tesseract dataset. Please note that you have to parametrize the names of the files opened, to load the right file.
    """
    num_features_options = ('all', '10k', '1k')
    assert num_features in num_features_options, f"num_features must be one of {num_features_options}."

    dataset_mode = X_fname.split('extended-features-X-')[-1].split('.json')[0]
    vec_path = join(ds_root_path, f"vec-{dataset_mode}.pkl")
    ds_path = join(ds_root_path, f"dataset-{dataset_mode}.pkl")

    if not os.path.exists(ds_path):
        print(f'Loading and processing dataset...')
        with open(join(ds_root_path, X_fname), 'r') as f:
            X = json.load(f)

        print('Loading labels...')
        with open(join(ds_root_path, y_fname), 'rt') as f:
            y = json.load(f)

        print('Loading timestamps...')
        with open(join(ds_root_path, meta_fname), 'rt') as f:
            T = json.load(f)
        T = [o['dex_date'] for o in T]
        T = np.array([datetime.datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') if "T" in o
                      else datetime.datetime.strptime(o, '%Y-%m-%d %H:%M:%S') for o in T])

        # Convert to numpy array and get feature names
        vec = DictVectorizer()
        X = vec.fit_transform(X).astype("float32")
        y = np.asarray(y)
        feature_names = vec.get_feature_names_out()

        my_save(vec, vec_path)

        # Get time index of each sample for easy reference
        time_index = {}
        for i in range(len(T)):
            t = T[i]
            if t.year not in time_index:
                time_index[t.year] = {}
            if t.month not in time_index[t.year]:
                time_index[t.year][t.month] = []
            time_index[t.year][t.month].append(i)

        data = (X, y, time_index, feature_names, T)
        data_names = ('X', 'y', 'time_index', 'feature_names', 'T')
        data_dict = {k: v for k, v in zip(data_names, data)}
        my_save(data_dict, ds_path)
    else:
        data_dict = my_load(ds_path)

    return data_dict


if __name__ == '__main__':
    data_dict = load_drebin_dataset()
    X, y, time_index, feature_names, T = tuple(data_dict.values())
    print("")