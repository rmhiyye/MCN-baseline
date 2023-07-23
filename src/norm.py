from transformers import AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_distances
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import pickle
import json
from itertools import chain

from src.model import Bertembedding

class n2c2Trainer(object):
        def __init__(self, config, test_dataset):
                self.config = config
                self.max_length = config.max_length
                self.embbed_size = config.embbed_size
                self.checkpoint_dir = config.checkpoint_dir
                self.num_training_steps = None

                # Load models
                self.device = "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.bertemb = Bertembedding(config.model_name).to(self.device)

                self.test_dataset = test_dataset

        def inference(self):

                self.bertemb.eval()

                # Load cui_encode dictionary
                with open('embedding_matrix/cui_encode.pickle', 'rb') as f:
                        cui_encode = pickle.load(f) 

                # Returns a dictionary of candidates ranked by their cosine similarity.
                dd_predictions, mentions_list = self.candidates_rank(self.test_dataset, cui_encode)
                print("Done.\n")

                del cui_encode
                return dd_predictions, mentions_list
        
        # create a function to output a dictionary of candidates ranked by their cosine similarity
        def candidates_rank(self, mentions_embeded, cui_encode):

                with open ('embedding_matrix/MENVectorMatrix.plk', 'rb') as f:
                        MENVectorMatrix = pickle.load(f)

                with open ('embedding_matrix/CUIVectorMatrix.plk', 'rb') as f:
                        CUIVectorMatrix = pickle.load(f)

                with open ('embedding_matrix/mentions_list.json', 'r') as f:
                        mentions_list = json.load(f)

                with open ('embedding_matrix/cui_keys.json', 'r') as f:
                        cui_keys = json.load(f)

                # Calculate distance matrix
                scoreMatrix = cosine_distances(MENVectorMatrix, CUIVectorMatrix)

                # Prepare prediction dictionary
                dd_predictions = {id: {'first candidate': [], 'top 5 candidates': []} for id in range(len(MENVectorMatrix))}

                # For each mention, find back the nearest cui vector, then attribute the associated cui:
                for i, id in enumerate(dd_predictions.keys()):
                        min_indices_10 = np.argpartition(scoreMatrix[i], 5)[:5]
                        min_indices_10 = min_indices_10[np.argsort(scoreMatrix[i][min_indices_10])]
                        # min_indices = min_indices_10[0]
                        min_indices = np.argmin(scoreMatrix[i])
                        
                        # Store the closest CUI in the predictions dictionary.
                        dd_predictions[id]['first candidate'] = [cui_keys[min_indices]]
                        dd_predictions[id]['top 5 candidates'] = [cui_keys[idx] for idx in min_indices_10]
                        
                return dd_predictions, mentions_list
        

        def tokenize(self, sentence):
                return self.tokenizer.encode_plus(sentence, padding="max_length", max_length=self.max_length, truncation=True, add_special_tokens=True, return_tensors="pt").to(self.device) # Tokenize input into ids.

        # Constructs two dictionnaries containing tokenized mentions (X) and associated labels (Y) respectively.
        def encoder(self, dataset, ref):
                X = dict()
                y = dict()
                cui = dict()
                with torch.no_grad():
                        for i, idx in tqdm(enumerate(dataset.keys()), total=len(dataset)):
                                cui[i] = dataset[idx]['cui']
                                X[i] = self.bertemb(self.tokenize(dataset[idx]['mention']))
                                y[i] = self.bertemb(self.tokenize(ref[cui[i]]))
                        nbMentions = len(X.keys())
                print("Number of mentions:", nbMentions)
                return X, y, cui

        # Mapping mentions
        # type and shape of output: numpy.ndarray, (nbMentions, embbed_size)
        def map_mention(self, dataset):
                print("-------------------------------")
                print("Mapping mentions in testing set...")
                mentions_embeded = np.zeros((len(dataset.keys()), self.embbed_size))
                mention = []
                with torch.no_grad():
                        for i, idx in tqdm(enumerate(dataset.keys()), total=len(dataset)):
                                mention.append(dataset[idx]['mention'])
                                tokenized_mention = self.tokenize(dataset[idx]['mention']).to(self.device)
                                mentions_embeded[i] = self.bertemb(tokenized_mention).cpu().numpy()

                return mentions_embeded, mention # Returns a matrix containing the embedding of each mention in the dataset