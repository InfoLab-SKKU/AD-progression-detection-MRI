#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import utilities as UT
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import csv
from collections import Counter
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
GPU = 2


# In[2]:


normal_scan_paths = glob.glob('/home/ubuntu/nasir/pytorch_3d_resnet_data/two_steps/1.CN/*.*')
abnormal_scan_paths = glob.glob('/home/ubuntu/nasir/pytorch_3d_resnet_data/two_steps/2.AD/*.*')
all_pths = normal_scan_paths + abnormal_scan_paths

standardize_features = '/home/ubuntu/nasir/pytorch_3d_resnet_data/two_steps_with_oversampling/standardized_features.csv'


df = pd.read_csv(standardize_features)
features = []
for indx, pth in enumerate(all_pths):
    split_str = pth.split('/')[-1].split('_MR')[0].split('_')[2:]
    sub_num = '_'.join(e for e in split_str)
#     print(sub_num)
    
    for ind, pid in enumerate(df['Subject_id'].to_list()):
        if sub_num == pid:
            entire_row = df.loc[ind]
            features.append(entire_row)

all_labels = []
for p in all_pths:
    if '1.CN' in p:
        all_labels.append(0)
    if '2.AD' in p:
        all_labels.append(1)

class Dataset_Early_Fusion(Dataset):
    def __init__(self, df, indices):
       
        self.paths = df['path']
        self.label = df['label']        
        
        self.all_features_df = pd.read_csv(standardize_features)
        
        self.all_features_df = self.all_features_df.fillna(0)
        
        self.all_features_df = self.all_features_df.iloc[indices]
        
        columns_to_drop = ['Unnamed: 0', 'AGE', 'PTGENDER', 'PTEDUCAT', 'Subject_id']       
        
        
        self.features_df = self.all_features_df.drop(columns = columns_to_drop)
           
    def __len__(self):
        return len(self.paths)
    
    
    def __getitem__(self,idx):     

        features_row = self.features_df.iloc[[idx]].values.tolist()      
        volume = np.load(self.paths[idx])  
        
        label = self.label[idx]     
        return volume, torch.tensor(features_row), int(label) 
    


# In[3]:


class RNN(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers * 2
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(0.7)        

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.dropout_layer(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_classes=2, input_shape=(1,110,110,110)): 
        super(ResNet3D, self).__init__()
        #stage 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(
            in_channels=input_shape[0],        
            out_channels=32,       
            kernel_size=(3,3,3),         
            padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(
            in_channels=32,       
            out_channels=32,      
            kernel_size=(3,3,3),          
            padding=1              
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),                  
            nn.Conv3d(
            in_channels=32,       
            out_channels=64,       
            kernel_size=(3,3,3), 
            stride=2,
            padding=1              
            )
        )
        #stage 2
        self.bot2=Bottleneck(64,64,1)
        #stage 3
        self.bot3=Bottleneck(64,64,1)
        #stage 4
        self.conv4=nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(
            in_channels=64,        
            out_channels=64,       
            kernel_size=(3,3,3),         
            padding=1,
            stride=2
            )
        )
        #stage 5
        self.bot5=Bottleneck(64,64,1)
        #stage 6
        self.bot6=Bottleneck(64,64,1)
        #stage 7
        self.conv7=nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(
            in_channels=64,        
            out_channels=128,       
            kernel_size=(3,3,3),          
            padding=1,
            stride=2
            )
        )
        #stage 8
        self.bot8=Bottleneck(128,128,1)
        
        #stage 9
        self.bot9=Bottleneck(128,128,1)
        
        #stage 10
        self.conv10=nn.Sequential(
            nn.MaxPool3d(kernel_size=(7,7,7)))
        
        self.feature_dimension = 1024
        self.sequence_length = 2
        self.hidden_size = 512
        self.num_layers = 1
               
        self.rnn = RNN(self.feature_dimension, 
                       self.sequence_length, 
                       self.hidden_size, 
                       self.num_layers, 
                       num_classes)
        
               
        self.dense_layer1 = nn.Sequential(
             nn.Linear(14, 32),
             nn.ReLU(),
             nn.Linear(32, 16),
             nn.ReLU())
             
        self.dense_layer2 = nn.Sequential(nn.Linear((512 * 2) * 2, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(256, 128),
                                 nn.ReLU())
        
        self.out = nn.Linear(16 + 128, 2)
        
        
    def forward(self, x, y, f):
        x = self.conv1(x)
        y = self.conv1(y)
        #print(x.shape)  
        x = self.bot2(x)
        y = self.bot2(y)
        #print(x.shape)
        x = self.bot3(x)
        y = self.bot3(y)
        #print(x.shape)
        x = self.conv4(x)
        y = self.conv4(y)
        #print(x.shape) 
        x = self.bot5(x)
        y = self.bot5(y)
        #print(x.shape)
        x = self.bot6(x)
        y = self.bot6(y)
        #print(x.shape)
        x = self.conv7(x)
        y = self.conv7(y)
        #print(x.shape)        
        x = self.bot8(x)
        y = self.bot8(y)
        #print(x.shape) 
        x = self.bot9(x)
        y = self.bot9(y)
        #print(x.shape)
        x = self.conv10(x)
        y = self.conv10(y)
        #print(x.shape) 
        x = x.view(x.size(0), -1) 
        y = y.view(y.size(0), -1)
        concat = torch.cat((x, y), dim=1)
        
        sequences = concat.reshape(-1, 2, 1024)
        out = self.rnn(sequences)
        out = self.dense_layer2(out)
        
        dens_output = self.dense_layer1(f)     
        
        cat_features = torch.cat((out, dens_output), 1)        
        probs = self.out(cat_features)      
        
        return probs


# In[4]:


def train(train_dataloader, val_dataloader, fold):
    net = ResNet3D().to(device)
    
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0)
#     opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma= 0.985)
#     scheduler = torch.optim.lr_scheduler.CyclicLR(opt, 
#                                                   base_lr=LR, 
#                                                   max_lr=0.001, 
#                                                   step_size_up=100,
#                                                   cycle_momentum=False)
    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))
        
    t = trange(EPOCHS, desc = ' ', leave=True)

    train_hist = []
    val_hist = []
    pred_result = []
    old_acc = 0
    old_auc = 0
    test_acc = 0
    best_epoch = 0
    test_performance=[]
    for e in t:    
        y_true = []
        y_pred = []
        
        val_y_true = []
        val_y_pred = []                
        
        train_loss = 0
        val_loss = 0

        # training
        net.train()
        for step, (img, features, label) in enumerate(train_dataloader):            
            
            img1 = img[:, :, :, 0:110]
            img2 = img[:, :, :, 110:]
            
            img1 = img1.unsqueeze(1)
            img2 = img2.unsqueeze(1)

            
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            
            features = features.squeeze(1).to(device)
            
            label = label.long().to(device)
            opt.zero_grad()
            out = net(img1, img2, features)            
            loss = loss_fcn(out, label)
            loss.backward()
            opt.step()
            
            label = label.cpu().detach()
            out = out.cpu().detach()
            y_true, y_pred = UT.assemble_labels(step, y_true, y_pred, label, out)        

            train_loss += loss.item()            

        train_loss = train_loss/(step+1)      
        
        acc = float(torch.sum(torch.max(y_pred, 1)[1]==y_true))/ float(len(y_pred))        
        auc = metrics.roc_auc_score(y_true, y_pred[:,1])        
        f1 = metrics.f1_score(y_true, torch.max(y_pred, 1)[1])
        precision = metrics.precision_score(y_true, torch.max(y_pred, 1)[1])
        recall = metrics.recall_score(y_true, torch.max(y_pred, 1)[1])
        ap = metrics.average_precision_score(y_true, torch.max(y_pred, 1)[1]) #average_precision

        # val
        net.eval()
        full_path = []
        with torch.no_grad():
            for step, (img, features, label) in enumerate(val_dataloader):
                                
                img1 = img[:, :, :, 0:110]
                img2 = img[:, :, :, 110:]

                img1 = img1.unsqueeze(1)
                img2 = img2.unsqueeze(1)

                img1 = img1.float().to(device)
                img2 = img2.float().to(device)

                features = features.squeeze(1).to(device)               
                
                label = label.long().to(device)
                out = net(img1, img2, features)
                
                loss = loss_fcn(out, label)
                val_loss += loss.item()

                label = label.cpu().detach()
                out = out.cpu().detach()
                val_y_true, val_y_pred = UT.assemble_labels(step, val_y_true, val_y_pred, label, out)                
                
                
        val_loss = val_loss/(step+1)
        val_acc = float(torch.sum(torch.max(val_y_pred, 1)[1]==val_y_true))/ float(len(val_y_pred))
        val_auc = metrics.roc_auc_score(val_y_true, val_y_pred[:,1])
        val_f1 = metrics.f1_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_precision = metrics.precision_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_recall = metrics.recall_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_ap = metrics.average_precision_score(val_y_true, torch.max(val_y_pred, 1)[1]) 


        train_hist.append([train_loss, acc, auc, f1, precision, recall, ap])
        val_hist.append([val_loss, val_acc, val_auc, val_f1, val_precision, val_recall, val_ap])             

        t.set_description("Epoch: %i, train loss: %.4f, train acc: %.4f, val loss: %.4f, val acc: %.4f, test acc: %.4f" 
                          %(e, train_loss, acc, val_loss, val_acc, test_acc))

        if(old_acc<val_acc):
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred            

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true))/ float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
            test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1]) #average_precision
            
        if(old_acc==val_acc) and (old_auc<val_auc):
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred           

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true))/ float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
            test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1]) #average_precision
            
            save_path = '/home/ubuntu/nasir/pytorch_3d_resnet_data/scripts/Repeated_Experiments/resnet_two_steps_cognitive_' + str(fold)
            torch.save(net.state_dict(), save_path)
            
            test_performance = [best_epoch, test_loss, test_acc, test_auc, test_f1, test_precision, test_recall, test_ap]
    return train_hist, val_hist, test_performance, test_y_true, test_y_pred


# In[5]:


TRAIN_BATCH_SIZE = 5
VALID_BATCH_SIZE = 5
EPOCHS = 120
LR = 0.000027
LOSS_WEIGHTS = torch.tensor([1., 1.]) 
device = torch.device('cuda:' + str(GPU) if torch.cuda.is_available() else 'cpu')


# In[ ]:


train_hist = []
val_hist = []
test_performance = []
test_y_true = np.asarray([])
test_y_pred = np.asarray([])
full_path = np.asarray([])


skf =  StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
for indx, (train_index, test_index) in enumerate(skf.split(all_pths, all_labels)):  
  
    X_train = [all_pths[p] for p in train_index]
    X_train_label = [all_labels[p] for p in train_index]
    
    X_test = [all_pths[p] for p in test_index]
    X_test_label = [all_labels[p] for p in test_index]   
   
    
    train_path = '/home/ubuntu/nasir/pytorch_3d_resnet_data/one_step/Train_Fold_single_cog_sing3' + str(indx) + '.csv'
    print('train path', train_path)
    
    csv_file = open(train_path, "w")
    writer = csv.writer(csv_file)
    for key, value in zip(X_train, X_train_label):
        writer.writerow([key, value])
    csv_file.close()
      
    valid_path = '/home/ubuntu/nasir/pytorch_3d_resnet_data/one_step/Valid_Fold_single_cog_sing3' + str(indx) + '.csv'
    csv_file = open(valid_path, "w")
    writer = csv.writer(csv_file)
    for key, value in zip(X_test, X_test_label):
        writer.writerow([key, value])
    csv_file.close()

    traindf = pd.read_csv(train_path,dtype=str)
    traindf.columns = ["path", "label"]
    validdf = pd.read_csv(valid_path,dtype=str)
    validdf.columns = ["path", "label"]
   
    train_ = Dataset_Early_Fusion(traindf, train_index)
    train_dataloader = torch.utils.data.DataLoader(train_, num_workers=2,
                                                   batch_size= TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

    valid = Dataset_Early_Fusion(validdf, test_index)
    valid_dataloader = torch.utils.data.DataLoader(valid, num_workers=2,
                                                   batch_size= VALID_BATCH_SIZE, shuffle=True, drop_last=True)

    cur_result = train(train_dataloader, valid_dataloader, indx)
    
    train_hist.append(cur_result[0])
    val_hist.append(cur_result[1]) 
    test_performance.append(cur_result[2]) 
    test_y_true = np.concatenate((test_y_true, cur_result[3].numpy()))
    
    if(len(test_y_pred) == 0):
        test_y_pred = cur_result[4].numpy()
    else:
        test_y_pred = np.vstack((test_y_pred, cur_result[4].numpy()))
#     full_path = np.concatenate((full_path, np.asarray(cur_result[5])))
    print('finish')

print('test_performance :', test_performance)

test_y_true = torch.tensor(test_y_true)
test_y_pred = torch.tensor(test_y_pred)

test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true.long()))/ float(len(test_y_pred))
test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1])

print('ACC %.4f, AUC %.4f, F1 %.4f, Prec %.4f, Recall %.4f, AP %.4f' 
      %(test_acc, test_auc, test_f1, test_precision, test_recall, test_ap))


# In[ ]:


z = {'net prediction: ' : test_y_pred,
     'true labels: ' : test_y_true}

np.save('/home/ubuntu/nasir/pytorch_3d_resnet_data/scripts/Repeated_Experiments/resnet-Two_Steps_cognitive.npy', 
       z)

def mean_std(net_pred, y_true):    

    s = 0
    e = 112
    all_acc = []
    all_auc = []
    all_f1 = []
    all_precision = []
    all_recall = []

    for i in range(1, 6, 1):

        net_pred = test_y_pred[s : e]
        y_true = test_y_true[s : e]

        all_acc.append( float(torch.sum(torch.max(net_pred, 1)[1]==y_true.long()))/ float(len(net_pred)) )
        all_auc.append(metrics.roc_auc_score(y_true, net_pred[:,1]))
        all_f1.append(metrics.f1_score(y_true, torch.max(net_pred, 1)[1]))

        all_precision.append( metrics.precision_score(y_true, torch.max(net_pred, 1)[1]) )
        all_recall.append( metrics.recall_score(y_true, torch.max(net_pred, 1)[1]) )

        s = s + 112
        e = s + 112


    all_acc = np.asarray(all_acc)
    all_auc = np.asarray(all_auc)
    all_f1 = np.asarray(all_f1)
    all_precision = np.asarray(all_precision)
    all_recall = np.asarray(all_recall)
    
    print('Average Accuracy: ', np.mean(all_acc), np.std(all_acc))
    print('Average AUC: ', np.mean(all_auc), np.std(all_auc))
    print('Average F1: ', np.mean(all_f1), np.std(all_f1))
    print('Average Precision: ', np.mean(all_precision), np.std(all_precision))
    print('Average Recall: ', np.mean(all_recall), np.std(all_recall))
    
mean_std(test_y_pred, test_y_true.long())


# In[8]:




