#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:12:59 2024

@author: vishalr
"""

import torch
import numpy as np
import pandas as pd
import os
from chr_dataset import CHRDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt



seed = 42
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)

device = torch.device('cuda:1')

tag = 'stamp0603'

rsam = CHRDataset(root="data", name=tag, use_node_attr=True, use_edge_attr=False)
train_dataset, val_dataset, test_dataset = random_split(rsam, [60000, 10000, 10000], generator=torch.Generator().manual_seed(seed))


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html

class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_in, dim_h, dim_out, dropout=0.2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, dim_out)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, batch):
        # Node embeddings 
        h = self.conv1(x, edge_index, edge_weight)
        h = h.relu()
        h = self.conv2(h, edge_index, edge_weight)
        h = h.relu()
        h = self.conv3(h, edge_index, edge_weight)

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=self.dropout, training=self.training)
        h = self.lin(h)
        
        return h


class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_in, dim_h, dim_out, dropout=0.2):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dim_in, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dim_out)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h1 = self.lin1(h)
        h1 = h1.relu()
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = self.lin2(h1)
        
        return h1
    
train_loss_rec = []
val_loss_rec = []    
#cmat = np.zeros((14, 14))

def train(model, train_loader, val_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    epochs = 120
    max_accuracy = 0
    
    model.train()
    for epoch in range(epochs + 1):
        train_loss = 0
        train_acc = 0
        sample_size = len(train_loader.dataset)
        
        # Train on batches
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            train_loss += loss.item()
            _, pred = out.max(dim=1)
            train_acc += pred.eq(data.y).sum().item()

            loss.backward()
            optimizer.step()
        
        if(epoch<50):
            scheduler.step()

        train_loss = train_loss / len(train_loader)
        train_loss_rec.append(train_loss)
        train_acc = train_acc / sample_size
        
        # Validation
        val_loss, val_acc = val(model, val_loader)
        val_loss_rec.append(val_loss)
        if(max_accuracy<val_acc):
            best_model = model
            max_accuracy = val_acc
        # Print metrics every 10 epochs
        if epoch % 1 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {train_loss:.2f} '
                  f'| Train Acc: {train_acc*100:>5.2f}% '
                  f'| Val Loss: {val_loss:.2f} '
                  f'| Val Acc: {val_acc*100:.2f}%')
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    
    plt.plot(train_loss_rec)
    plt.plot(val_loss_rec)
    #plt.savefig('/data/shared/vishal/rSAM/gnn_graphs_1/data/rsam_'+tag+'/loss.png')
    test_loss, test_acc = test(model, test_loader)
    #print(cmat)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    return best_model


@torch.no_grad()
def val(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    sample_size = len(loader.dataset)
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y).item()
        _, pred = out.max(dim=1)
        acc += pred.eq(data.y).sum().item()
    return loss / len(loader), acc / sample_size


@torch.no_grad()
def test(model, loader):
    print('###test')
    out_embed = []
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    sample_size = len(loader.dataset)
    
    for data in loader:
        data = data.to(device)
        #print('seven')
        #print(data.edge_attr)
        #print(data.x, data.edge_index, data.batch, data.uid)
        out = model(data.x, data.edge_index, data.batch)
        #print(out.cpu)
        out_embed.extend(out.tolist())
        #print(len(out_embed))
        #print(data.uid)
        #print(out.argmax(1))
        #print(data.y)
        
        loss += criterion(out, data.y).item()
        _, pred = out.max(dim=1)
        acc += pred.eq(data.y).sum().item()
        
        #break
    #print(uid_known, uid_pred)
    #cmat = confusion_matrix(uid_known, uid_pred)
    
    
    
    
    
    #gnn_files_loc = '/data/shared/vishal/rSAM/gnn_graphs_1/data/rsam_'+tag+'/'
    #with open(os.path.join(gnn_files_loc,'rsam_uid_map.txt'), 'r') as fw:
    #    uid_map = fw.read().splitlines()
      
    #with open(os.path.join('/data/shared/vishal/rSAM/gnn_graphs_1/data/rsam_'+tag+'/results/','gin_predictions.txt'), 'w') as fw:
    #    for itr, item in enumerate(uid_list):
            #print(itr, uid_list[itr])
    #        fw.write(str(uid_map[uid_list[itr]]) + '\n')
            
    
            
   
    return loss / len(loader), acc / sample_size

#gcn = GCN(dim_in=rsam.num_node_features, dim_h=32, dim_out=rsam.num_classes).to(device)
gin = GIN(dim_in=rsam.num_node_features, dim_h=32, dim_out=rsam.num_classes).to(device)
#gcn, cmat = train(gcn, train_loader, val_loader, test_loader)
gin = train(gin, train_loader, val_loader, test_loader)


#torch.save(gin.state_dict(), os.path.join('/data/shared/vishal/rSAM/gnn_graphs_1/data/rsam_'+tag+'/results/','gin_model.pth'))
'''
model = GIN(dim_in=rsam.num_node_features, dim_h=32, dim_out=rsam.num_classes).to(device)
model.load_state_dict(torch.load(os.path.join('/data/shared/vishal/rSAM/gnn_graphs_1/data/rsam_'+tag+'/results/','gin_model3.pth')))
model.eval()
test_loss, test_acc, cmat = test(model, test_loader)
'''