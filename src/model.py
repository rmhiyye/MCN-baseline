####################
# addition of layers for fine-tuning bert;
# activate function
####################

import torch
from torch import nn
from transformers import AutoModel

# bert model
class Bertembedding(nn.Module):
    def __init__(self, model_name):
        super(Bertembedding, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, tokens):
        tokens = {key: value.squeeze(1) for key, value in tokens.items()}
        bert_output = self.bert(**tokens)[0][:,0,:]
        x = torch.nn.functional.normalize(bert_output)
        return x

# linear layer
class NeuralNetwork(nn.Module):
    def __init__(self, embbed_size, linear_size):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(embbed_size, out_features=linear_size) # Single linear layer
        torch.nn.init.eye_(self.linear.weight) # Linear layer weights initialization

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        x = self.linear(x)
        return x

# concatenate bert and linear layer
class BertWithCustomNNClassifier(nn.Module):
    """
    A pre-trained BERT model with a custom classifier.
    The classifier is a neural network implemented in this class.
    """
    
    def __init__(self, model_name, linear_size):
        super(BertWithCustomNNClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(in_features=768, out_features=linear_size) # TODO: automatically get the size of the last layer of BERT (768) instead of hardcoding it
        torch.nn.init.eye_(self.linear1.weight)

        
    def forward(self, tokens):
        tokens = {key: value.squeeze(1) for key, value in tokens.items()}
        # bert_output = self.bert(**tokens)[0][:,0,:]
        # x = torch.nn.functional.normalize(bert_output)
        bert_output = self.bert(**tokens)[0]
        avg_output = torch.mean(bert_output, dim=1)
        x = torch.nn.functional.normalize(avg_output)
        x = self.linear1(x)
        return x
        
    def freeze_bert(self):
        """
        Freezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.bert.parameters():
            param.requires_grad=False
    
    def unfreeze_bert(self):
        """
        Unfreezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        both the wieghts of the custom classifier and of the underlying BERT are modified.
        """
        for param in self.bert.parameters():
            param.requires_grad=True
