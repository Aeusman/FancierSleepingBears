import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def default_args():
    return None

class SeqCNNModel(nn.Module):
    
    def __init__(self, args):
        super(SeqCNNModel, self).__init__()

        # initialize module parameters
        self.embed = nn.Embedding(args["vocab_size"], args["embed_size"])
        self.num_layers = args["num_layers"]
        self.conv = nn.ModuleList([nn.Conv1d(args["embed_size"], args["embed_size"], args["kernel_sizes"][0]) for k in range(self.num_layers)])
        
        self.maxpool = nn.MaxPool1d(args["kernel_sizes"][0])
        self.dropout = nn.Dropout(args["dropout"])
        self.output = nn.Linear(args["chout"], args["num_classes"])

    def forward(self, x):

        x = self.embed(x)
        for i in range(self.num_layers):
            x = self.conv[i](x)
            x = F.relu(x)
            x = self.maxpool(x)
            x = self.dropout(x)
        x = self.output(x)
        return F.LogSoftmax(x)

class ParCNNModel(nn.Module):

    def __init__(self, args):
        super(ParCNNModel, self).__init__()
        
        self.embed = nn.Embedding(args["vocab_size"], args["embed_size"])
        self.conv = nn.ModuleList([nn.Conv1d(args["embed_size"], args["embed_size"], k) for k in args["kernel_sizes"]])

        self.maxpool = nn.MaxPool1d(args["kernel_sizes"][0])
        self.dropout = nn.Dropout(args["dropout"])
        self.output = nn.Linear(args["embed_size"] * len(args["kernel_sizes"]), args["num_classes"])

    def forward(self, x):

        x = self.embed(x)
        x = [self.maxpool(self.relu(conv(x))) for conv in self.conv]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.output(x)
        retrun F.LogSoftmax(x)
