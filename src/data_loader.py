####################
# 2019-n2c2-MCN dataset loader:
####################

import os
import random
import torch
from torch.utils.data import DataLoader
import umls_api
from requests.exceptions import HTTPError

from src import config
from src.utils import lowercaser_mentions, id_combination

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, cui_dict):
        self.mention = X
        self.label = y
        self.cui = cui_dict

    def __getitem__(self, idx):
        mention = self.mention[idx]
        label = self.label[idx]
        cui = self.cui[idx]
        sample = (mention, label, cui)
        return sample

    def __len__(self):
        return len(self.mention)
    
# loading the dataset
def data_loader():

    path = config.data_path

    train_path = os.path.join(path, 'train/train_file_list.txt')
    test_path = os.path.join(path, 'test/test_file_list.txt')
    train_file = file_list_loader(train_path)
    test_file = file_list_loader(test_path)

    train_ref_path = os.path.join(path, 'train/cui2name_v2.txt')
    test_ref_path = os.path.join(path, 'umls_cui.txt')
    # train_ref_dataset = cui2name(train_ref_path)
    # test_ref_dataset = cui2name(test_ref_path)

    ##########################################################################
    # 1/4 of training set
    ##########################################################################
    random.seed(config.seed)
    selected_keys = random.sample(list(train_file.keys()), len(train_file)//4)
    train_small_file = {k: train_file[k] for k in selected_keys}
    ##########################################################################
    # 1/4 of test set
    ##########################################################################
    # selected_keys = random.sample(list(test_file.keys()), len(test_file)//4)
    # test_small_file = {k: test_file[k] for k in selected_keys}

    train_note_path, train_norm_path = os.path.join(path, 'train/train_note'), os.path.join(path, 'train/train_norm')
    test_note_path, test_norm_path = os.path.join(path, 'test/test_note'), os.path.join(path, 'gold/test_norm')

    
    train_dataset, train_cui_less_dict, train_span_split = mention2concept(train_note_path, train_norm_path, train_small_file, with_text = False)
    test_dataset, test_cui_less_dict, test_span_split = mention2concept(test_note_path, test_norm_path, test_file, with_text = False)    

    # pre-processing
    train_dataset = id_combination(train_dataset)
    train_dataset = lowercaser_mentions(train_dataset)
    
    test_dataset = id_combination(test_dataset)
    test_dataset = lowercaser_mentions(test_dataset)

    # split test_dataset into test_dataset and validation_dataset (1:1)
    selected_keys = random.sample(list(test_dataset.keys()), len(test_dataset)//2)
    validation_dataset = {k: test_dataset[k] for k in selected_keys}
    # test_dataset = {k: test_dataset[k] for k in test_dataset.keys() if k not in selected_keys}

    return train_dataset, test_dataset, validation_dataset, train_cui_less_dict, train_span_split, test_span_split

'''
Create empty file list dictionary
'''
def file_list_loader(file_list_path): # ./dataset/train/train_file_list.txt
    with open(f"{file_list_path}", "r") as fl:
        lines = fl.readlines()
    file_dict = dict()
    keys = []
    for line in lines:
        line = line.strip()
        keys.append(line)
    file_dict = dict.fromkeys(keys)
    return file_dict

def norm_list_loader(norm_list_path): # ./dataset/train/train_norm.txt
    with open(f"{norm_list_path}", "r") as fl:
        lines = fl.readlines()
    norm_list = []
    for line in lines:
        line = line.strip()
        norm_list.append(line)
    return norm_list

'''
file_dict:
    {"0034":
            {"N000":
                    {"cui": .. ,
                     "mention", ..},
             "N003":
             ...
             "text": '054478430 ELMVH\n79660638\n1835979\n12/11/2005..'},
     "0038":
     ...
     }
     
n_span_split:
    '2': norm_id || cui || start || end || start || end || start || end
    '1': norm_id || cui || start || end || start || end 
    '0': norm_id || cui || start || end 
'''
def mention2concept(note_path, norm_path, file_dict, with_text = True):
    n_span_split_keys = ["2", "1", "0"]
    n_span_split = {key: 0 for key in n_span_split_keys}
    cui_less_dict = dict()
    for key in file_dict.keys():
        sub_dict = dict()
        with open(f"{note_path}/{key}.txt", "r") as fl:
            texts = str(fl.read())
        with open(f"{norm_path}/{key}.norm", "r") as fl:
            lines = fl.readlines()
        for line in lines:
            subsub_dict = dict()
            line = line.strip()
            # norm_id, cui, start, end = line.split("||")
            line = line.split("||")
            norm_id = line[0]
            if line[1] != 'CUI-less':
                cui = line[1]
                line[2:] = [int(x) for x in line[2:]] # convert str into int
                span_split = round((len(line)-2)/2 - 1) # (number_terms - number_id_and_cui)/2 - 1
                n_span_split[str(span_split)] += 1
                mention = texts[line[2]: line[3]]
                for i in range(1, int(span_split)+1):
                    mention += texts[line[2+2*i]: line[3+2*i]]
                subsub_dict["cui"] = cui
                subsub_dict["mention"] = mention
                sub_dict[norm_id] = subsub_dict
            else:
                cui_less_dict[key] = 'CUI-less'
        if with_text:
            sub_dict["text"] = texts
        file_dict[key] = sub_dict
    return file_dict, cui_less_dict, n_span_split

def cui2name(file_path):
    cui2name_dict = dict()
    with open(f"{file_path}", "r") as fl:
        lines = fl.readlines()
    for line in lines:
        line = line.strip()
        cui = line.split("||")[0]
        cui_name = line.split("||")[1]
        if cui in cui2name_dict:
            cui2name_dict[cui].append(cui_name)
        else:
            cui2name_dict[cui] = [cui_name]
    return cui2name_dict

def get_cui_name(cui):
    key = config.api_key
    try:
        api = umls_api.API(api_key=key)
        name = api.get_cui(cui)['result']['name']
    except HTTPError:
        print(f"HTTPError occurred for CUI: {cui}")
        name = 'NAME-less'
    return name

def norm_list_generator(file_dict):
    norm_list = []
    for key in file_dict.keys():
        for idx in file_dict[key].keys():
            if file_dict[key][idx]['cui'] in norm_list:
                continue
            else:
                norm_list.append(file_dict[key][idx]['cui'])
    return norm_list