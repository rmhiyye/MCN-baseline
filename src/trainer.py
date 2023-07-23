from transformers import AutoTokenizer, AutoModel, get_scheduler
from torch.utils.data import DataLoader, RandomSampler
import torch
from scipy.spatial.distance import cdist
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
import faiss

import src.config as config
from src.data_loader import Dataset
from src.model import BertWithCustomNNClassifier, NeuralNetwork, Bertembedding
from src.utils import eval_map
torch.manual_seed(config.seed)

class n2c2Trainer(object):
        def __init__(self, args, train_dataset=None, test_dataset=None, validation_dataset=None, train_ref_dataset=None, test_ref_dataset=None):
                self.config = config
                self.args = args
                self.max_length = config.max_length
                self.embbed_size = config.embbed_size
                self.checkpoint_dir = config.checkpoint_dir
                self.num_training_steps = None

                # Load models
                self.device = "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.bertemb = Bertembedding(config.model_name).to(self.device)
                self.model = NeuralNetwork(config.embbed_size, config.embbed_size).to(self.device)
                
                if self.args.do_train:
                        # Load datasets
                        print("--------------------------")
                        print("Embedding training set...")
                        print("--------------------------")
                        X_train, y_train, cui_train = self.encoder(train_dataset, train_ref_dataset)
                        X_val, y_val, cui_val = self.encoder(validation_dataset, test_ref_dataset)
                        train_set = Dataset(X_train, y_train, cui_train)
                        val_set = Dataset(X_val, y_val, cui_val)
                        # Create a seeded RandomSampler
                        random_sampler_train = RandomSampler(train_set, replacement=False, generator=torch.Generator().manual_seed(config.seed))
                        random_sampler_val = RandomSampler(val_set, replacement=False, generator=torch.Generator().manual_seed(config.seed))
                        self.train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=random_sampler_train)
                        self.val_loader = DataLoader(val_set, batch_size=config.batch_size, sampler=random_sampler_val)
                        self.num_training_steps = config.epochs * len(self.train_loader)
                        # For MAP evaluation
                        # self.train_cui_encode = self.map_cui(train_ref_dataset)
                        # self.val_cui_encode = self.map_cui(test_ref_dataset)

                        # Optimizer
                        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)        
                        # Scheduler
                        self.lr_scheduler = get_scheduler(
                        name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps)
                        # Loss function
                        self.loss_fn = cos_dist
                        # Tensorboard
                        if self.args.save_tb:
                                self.writer = SummaryWriter()

                self.train_ref_dataset = train_ref_dataset
                self.test_ref_dataset = test_ref_dataset
                self.test_dataset = test_dataset

        def train(self):
                # self.basenorm.train()
                self.model.train()
                # Train
                start_time = time.time()
                for epoch in range(self.config.epochs):
                        epoch_loss = 0
                        train_map = 0
                        idx = 0
                        all_pred = []
                        all_cui = dict()
                        for X, y, CUI in self.train_loader: # both X and y contains n=batch_size tokenized mentions and labels respectively
                                batch_loss = None
                                pred = self.model(X)
                                ground_truth = self.model(y)
                                loss = self.loss_fn(pred, ground_truth) # Cosine similarity between embedding of mention and associated label.
                                all_pred.append(pred.cpu().detach().numpy())
                                for cui in CUI:
                                        all_cui[idx] = cui
                                        idx += 1

                                # Backpropagation
                                self.optimizer.zero_grad()
                                batch_loss = torch.mean(loss) # Averages loss over the whole batch.
                                batch_loss.backward()
                                self.optimizer.step()
                                self.lr_scheduler.step()

                                epoch_loss += batch_loss.item()

                        self.model.eval()
                        all_pred = np.concatenate(all_pred, axis=0)
                        # self.train_cui_encode = self.map_cui(self.train_ref_dataset)
                        train_pred = self.candidates_rank(all_pred, self.train_cui_encode)
                        train_map = eval_map(all_cui, train_pred)
                        # evaluate loss and MAP(k=1,5) in each epoch
                        # val_loss, val_map = self.evaluate()
                        self.model.train()

                        # Tensorboard
                        if self.args.save_tb:
                                self.writer.add_scalar("Loss/train", epoch_loss/len(self.train_loader) , epoch)
                                self.writer.add_scalar("MAP/train", train_map , epoch)
                                self.writer.add_scalar("Loss/val", val_loss, epoch)
                                self.writer.add_scalar("MAP/val", val_map, epoch)

                        # print loss and MAP(k=1,5) for training set and validation set every * epochs
                        if (epoch + 1) % self.config.show_step == 0:
                                print("-------------------------------")
                                print("Epoch [{}/{}], Elapsed Time: {:.2f}mins".format(epoch+1, self.config.epochs, (time.time() - start_time) / 60))
                                print(f"train_loss = {epoch_loss / len(self.train_loader)} | train_map = {train_map}")
                                print(f"val_loss = {val_loss} | val_map = {val_map}")

                        # Save model for * epochs
                        if (epoch + 1) % self.config.save_step == 0:
                                self.save_checkpoint(epoch, self.model, self.optimizer, batch_loss.item())

        def evaluate(self):
                all_pred = []
                all_cui = dict()
                idx = 0
                with torch.no_grad():
                        for X, y, CUI in self.val_loader: # both X and y contains n=batch_size tokenized mentions and labels respectively
                                batch_loss = None
                                pred = self.model(X)
                                ground_truth = self.model(y)
                                loss = self.loss_fn(pred, ground_truth) # Cosine similarity between embedding of mention and associated label.
                                all_pred.append(pred.cpu().detach().numpy())
                                for cui in CUI:
                                        all_cui[idx] = cui
                                        idx += 1
                                
                                batch_loss = torch.mean(loss) # Averages loss over the whole batch.
                        all_pred = np.concatenate(all_pred, axis=0)
                        self.val_cui_encode = self.map_cui(self.test_ref_dataset)
                        val_pred = self.candidates_rank(all_pred, self.val_cui_encode)
                        val_map = eval_map(all_cui, val_pred)
                        return batch_loss.item(), val_map

        # checkpoint saver function
        def save_checkpoint(self, epoch, model, optimizer, loss):
                # Check if the directory already exists
                if not os.path.exists(self.checkpoint_dir):
                # If the directory does not exist, create it
                    os.makedirs(self.checkpoint_dir)
                checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                            }
                # Save the checkpoint
                torch.save(checkpoint, f'{self.checkpoint_dir}/epoch{epoch+1}_checkpoint.pt')

        def inference(self):
                # epoch = input('please enter the epoch number of the checkpoint you want to load: ')
                # epoch = 1
                # checkpoint = torch.load(f'{self.checkpoint_dir}/epoch{epoch}_checkpoint.pt')
                # self.checkpoint_loader(checkpoint, epoch, for_inference=True)
                self.bertemb.eval()
                X_pred, mention = self.map_mention(self.test_dataset)

                # Load cui_encode dictionary
                with open('embedding_matrix/umls_encode.pickle', 'rb') as f:
                        cui_encode = pickle.load(f) 

                # cui = self.map_cui(cui_encode)

                # Returns a dictionary of candidates ranked by their cosine similarity.
                dd_predictions = self.candidates_rank(X_pred, cui_encode)
                print("Done.\n")

                del cui_encode
                return dd_predictions, mention
        
        # create a function to output a dictionary of candidates ranked by their cosine similarity
        def candidates_rank(self, X_pred, cui_encode):
                # Prepare prediction dictionary
                dd_predictions = {id: {'first candidate': [], 'top 5 candidates': []} for id in range(len(X_pred))}

                # Flatten the list of embeddings
                flat_embeddings = [embedding for embeddings in cui_encode.values() for embedding in embeddings]

                # Create the CUI vector matrix
                CUIVectorMatrix = np.squeeze(np.stack(flat_embeddings))

                # Calculate distance matrix
                scoreMatrix = cdist(X_pred, CUIVectorMatrix, 'cosine')

                cui_keys = []
                for cui, tensor_list in cui_encode.items():
                        cui_keys.extend([cui] * len(tensor_list))

                # For each mention, find back the nearest cui vector, then attribute the associated cui:
                for i, id in enumerate(dd_predictions.keys()):
                        min_indices_5 = np.argpartition(scoreMatrix[i], 5)[:5]
                        min_indices_5 = min_indices_5[np.argsort(scoreMatrix[i][min_indices_5])]
                        min_indices = min_indices_5[0]
                        
                        # Store the closest CUI in the predictions dictionary.
                        dd_predictions[id]['first candidate'] = [cui_keys[min_indices]]
                        dd_predictions[id]['top 5 candidates'] = [cui_keys[idx] for idx in min_indices_5]

                return dd_predictions

                
        def checkpoint_loader(self, checkpoint, epoch, for_inference=False):
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if for_inference == False:
                        epoch = checkpoint['epoch']
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        batch_loss = checkpoint['loss']
                        return epoch, batch_loss

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
        
        # Mapping ontology concepts
        def map_cui(self, ref):
                cui = dict()
                with torch.no_grad():
                        for cui_key, cui_syns in tqdm(ref.items()):
                                cui[cui_key] = []
                                for cui_syn in cui_syns:
                                        cui_syn_tensor = torch.from_numpy(cui_syn).to(self.device)
                                        cui[cui_key].append(self.model(cui_syn_tensor).cpu().numpy())
                return cui # Returns a dictionnary containing the embedding of each concept in the ontology.
        
        # Mapping mentions
        # type and shape of output: numpy.ndarray, (nbMentions, embbed_size)
        def map_mention(self, dataset):
                print("-------------------------------")
                print("Mapping mentions in testing set...")
                X_pred = np.zeros((len(dataset.keys()), self.embbed_size))
                mention = []
                with torch.no_grad():
                        for i, idx in tqdm(enumerate(dataset.keys()), total=len(dataset)):
                                mention.append(dataset[idx]['mention'])
                                tokenized_mention = self.tokenize(dataset[idx]['mention']).to(self.device)
                                # X_pred[i] = self.model(self.bertemb(tokenized_mention)).cpu().numpy()
                                X_pred[i] = self.bertemb(tokenized_mention).cpu().numpy()

                return X_pred, mention # Returns a matrix containing the embedding of each mention in the dataset.

def cos_dist(t1, t2):
    cos = nn.CosineSimilarity()
    cos_sim = 1 + cos(t1, t2)*(-1)
    return cos_sim