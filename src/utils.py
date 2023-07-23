####################
# id_combination, lowercaser_mentions
####################
from sklearn.metrics import f1_score
import src.config as config
from requests.exceptions import HTTPError
import umls_api

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def id_combination(norm_dict):
    '''
    input:
        {"0034":
            {"N000":
                {"cui": .. ,
                 "mention", ..}
            }
        }
    output:
        {"0034_N000":
            {"cui": .. ,
             "mention", ..}
        }
    '''
    combin_dict = dict()
    for file_id in norm_dict.keys():
        for norm_id in norm_dict[file_id].keys():
            combin_id = file_id + "_" +norm_id
            combin_dict[combin_id] = norm_dict[file_id][norm_id]
    return combin_dict

def lowercaser_mentions(train_dict):
    for key in train_dict.keys():
        train_dict[key]["mention"] = train_dict[key]["mention"].lower()
    return train_dict

def eval_accuracy(actual, predicted):
    acc = 0
    for i, key in enumerate(actual.keys()):
        if isinstance(actual[key], dict):
            true_cui = actual[key]['cui']
        else:
            true_cui = actual[key]
        pred_cui = predicted[i]['first candidate'][0]
        if true_cui == pred_cui:
            acc += 1
    return acc / len(actual.keys())

# calculate mean average precision (k=1, 5)
def eval_map(actual, predicted, mentions, k=1):
    aps = []

    len_dict = {key: [] for key in range(1, 7+1)}

    for i, key in enumerate(actual.keys()):
        if isinstance(actual[key], dict):
            true_cui = actual[key]['cui']
        else:
            true_cui = actual[key]
        pred_cui = predicted[i]['first candidate'] if k == 1 else predicted[i]['top 5 candidates']
        
        # mentions_len = len(mentions[i].split(' ')) if len(mentions[i].split(' ')) <= 4 else 4
        mentions_len = len(mentions[i].split(' ')) if len(mentions[i].split(' ')) <= 7 else 7

        num_relevant_items = 0
        sum_precisions = 0
        '''
        for j, pred in enumerate(pred_cui, start=1):
            if pred == true_cui:
                num_relevant_items += 1
                precision_at_j = num_relevant_items / j
                sum_precisions += precision_at_j

        ap = sum_precisions / num_relevant_items if num_relevant_items > 0 else 0
        aps.append(ap)
        '''
        for j, pred in enumerate(pred_cui, start=1):
            if pred == true_cui:
                num_relevant_items += 1
                break

        ap = num_relevant_items 
        aps.append(ap)
        len_dict[mentions_len].append(ap)

    map = sum(aps) / len(aps)
    len_map_dict = {
                    key: {'MAP': sum(len_dict[key]) / len(len_dict[key]), 'size': len(len_dict[key])}
                    if len(len_dict[key]) > 0 else 0
                    for key in len_dict.keys()
                }
    
    return map, len_map_dict


def get_cui_name(cui):
    key = config.api_key
    try:
        api = umls_api.API(api_key=key)
        name = api.get_cui(cui)['result']['name']
    except HTTPError:
        print(f"HTTPError occurred for CUI: {cui}")
        name = 'NAME-less'
    return name

def text_preprocessing(text):

    tokens = word_tokenize(text)

    stopwords_list = stopwords.words('english')
    stemmer = PorterStemmer()

    tokens_filtered = []

    for token in tokens:
        if token not in stopwords_list:
            token = stemmer.stem(token)
            tokens_filtered.append(token.lower())

    