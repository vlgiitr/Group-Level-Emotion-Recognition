#----------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------

# coding: utf-8

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

import matplotlib.pyplot as plt

#---------------------------------------------------------------------------
# IMPORTANT PARAMETERS
#---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else 'cpu'
root_dir = "Dataset/"
epochs = 10
batch_size = 32
maxFaces = 27

#---------------------------------------------------------------------------
# DATASET AND LOADERS
#---------------------------------------------------------------------------

neg_train = sorted(os.listdir('Dataset/emotiw/train/'+'Negative/'))
neu_train = sorted(os.listdir('Dataset/emotiw/train/'+'Neutral/'))
pos_train = sorted(os.listdir('Dataset/emotiw/train/'+'Positive/'))

neg_val = sorted(os.listdir('Dataset/emotiw/val/' + 'Negative/'))
neu_val = sorted(os.listdir('Dataset/emotiw/val/' + 'Neutral/'))
pos_val = sorted(os.listdir('Dataset/emotiw/val/' + 'Positive/'))

train_filelist = neg_train + neu_train + pos_train
val_filelist = neg_val + neu_val + pos_val
print(len(train_filelist),len(val_filelist))

for i in train_filelist:
    if i[0] != 'p' and i[0] != 'n':
        train_filelist.remove(i)

for i in val_filelist:
    if i[0] != 'p' and i[0] != 'n':
        val_filelist.remove(i)

dataset_sizes = [len(train_filelist), len(val_filelist)]
data_sizes = [[2756, 3077, 3975], [1231, 1366, 1745]]

train_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class EmotiWDataset(Dataset):
    
    def __init__(self, filelist, root_dir, loadTrain=True, transform=transforms.ToTensor()):
        """
        Args:
            filelist: List of names of image/feature files.
            root_dir: Dataset directory
            transform (callable, optional): Optional transformer to be applied
                                            on an image sample.
        """
        
        self.filelist = filelist
        self.root_dir = root_dir
        self.transform = transform
        self.loadTrain = loadTrain
            
    def __len__(self):
        return (len(self.filelist))   
    
    def __getitem__(self, idx):
        train = ''
        if self.loadTrain:
            train = 'train'
        else:
            train = 'val'
        filename = self.filelist[idx].split('.')[0]
        labeldict = {'neg':'Negative',
                     'neu':'Neutral',
                     'pos':'Positive',
                     'Negative': 0,
                     'Neutral': 1,
                     'Positive':2}

        labelname = labeldict[filename.split('_')[0]]
        image = Image.open(self.root_dir+'emotiw/'+train+'/'+labelname+'/'+filename+'.jpg')
        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            image_1 = np.zeros((3, 224, 224), dtype = float)
            image_1[0] = image
            image_1[1] = image
            image_1[2] = image
            image = image_1
            image = torch.FloatTensor(image.tolist())
        features = np.load(self.root_dir+'FaceFeatures/'+train+'/'+labelname+'/'+filename+'.npz')['a']
        numberFaces = features.shape[0]
        maxNumber = min(numberFaces, maxFaces)
        features1 = np.zeros((maxFaces, 256), dtype = 'float32')
        for i in range(maxNumber):
            features1[i] = features[i]
        features1 = torch.from_numpy(features1)
        sample = {'image': image, 'features': features1, 'label':labeldict[labelname], 'numberFaces': numberFaces}
        return sample


train_dataset = EmotiWDataset(train_filelist, root_dir, loadTrain = True, transform=train_data_transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

val_dataset = EmotiWDataset(val_filelist, root_dir, loadTrain=False, transform = val_data_transform)

val_dataloader = DataLoader(val_dataset, shuffle =True, batch_size = batch_size, num_workers = 0)

print('Dataset Loaded')

#---------------------------------------------------------------------------
# MODEL DEFINITION
#---------------------------------------------------------------------------

global_model = torch.load('./TrainedModels/DenseNet161_EmotiW').module.features

for param in global_model.parameters():
    param.requires_grad = False

class FaceAttention(nn.Module):
    def __init__(self, global_model):
        super(FaceAttention, self).__init__()
        
        self.global_model = global_model
        self.global_fc3_debug = nn.Linear(2464, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)
        self.bn_global = nn.BatchNorm1d(2208, affine=False)
        self.bn_face_features = nn.BatchNorm1d(256, affine=False)
        self.dropout_classifier = nn.Dropout(0.5)
    
    def forward(self, image, face_features, numberFaces):
        features = self.global_model.forward(image)

        out = F.relu(features, inplace=True)
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        global_features_initial = Variable(global_features_initial)
        batch_size = face_features.shape[0]
        global_features_initial = global_features_initial.view(-1,2208)
        face_features_sum = torch.sum(face_features, dim=1)
        face_features_sum = face_features_sum.view(-1, 256)
        for i in range(batch_size):
            faces_num_div = float(min(numberFaces[i], maxFaces))
            if faces_num_div != 0:
                face_features_sum[i] = face_features_sum[i] / faces_num_div
        #THE face_features_sum TENSOR NOW CONTAINS AVERAGE OF THE FACE FEATURES

        face_features_sum = self.bn_face_features(face_features_sum)
        global_features_initial = self.bn_global(global_features_initial)

        final_features = torch.cat((face_features_sum, global_features_initial), dim=1)
        final_features = self.dropout_classifier(final_features)

        x = (self.global_fc3_debug(final_features))
        return x

model_ft = FaceAttention(global_model)
model_ft = model_ft.to(device)
model_ft = torch.nn.DataParallel(model_ft)

params = list(model_ft.module.global_fc3_debug.parameters())

#---------------------------------------------------------------------------
# TRAINING
#---------------------------------------------------------------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in range(2):            
            if phase == 0:
                dataloaders = train_dataloader
                scheduler.step()
                model.train()
            else:
                dataloaders = val_dataloader
                model.eval()
            
            running_loss_neg = 0.0
            running_corrects_neg = 0
            running_loss_neu = 0.0
            running_corrects_neu = 0
            running_loss_pos = 0.0
            running_corrects_pos = 0
            running_loss = 0.0
            running_corrects = 0
            
            for i_batch, sample_batched in enumerate(dataloaders):
                inputs = sample_batched['image']
                labels = sample_batched['label']
                face_features = sample_batched['features']
                numberFaces = sample_batched['numberFaces']
                inputs = inputs.to(device)
                labels = labels.to(device)
                face_features = face_features.to(device)
                numberFaces = numberFaces.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 0):
                    outputs = model(inputs, face_features, numberFaces)
                    print("Forward "+ str(i_batch))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                            
                    labels_pos = np.where(labels == 2)
                    labels_neu = np.where(labels == 1)
                    labels_neg = np.where(labels == 0)
                    
                    outputs_pos = outputs[labels_pos]
                    outputs_neu = outputs[labels_neu]
                    outputs_neg = outputs[labels_neg]

                    if len(labels_pos[0]) != 0:
                        loss_pos = criterion(outputs_pos, labels[labels_pos])
                    else:
                        loss_pos = torch.FloatTensor([0])
                    
                    if len(labels_neu[0]) != 0:
                        loss_neu = criterion(outputs_neu, labels[labels_neu])
                    else:
                        loss_neu = torch.FloatTensor([0])
                        
                    if len(labels_neg[0]) != 0:
                        loss_neg = criterion(outputs_neg, labels[labels_neg])
                    else: 
                        loss_neg = torch.FloatTensor([0])
                    
                    if phase == 0:
                        loss.backward()
                        optimizer.step()
  
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                running_loss_pos += loss_pos.item() * inputs[labels_pos].size(0)
                running_corrects_pos += torch.sum(preds[labels_pos] == labels[labels_pos].data)
                running_loss_neu += loss_neu.item() * inputs[labels_neu].size(0)
                running_corrects_neu += torch.sum(preds[labels_neu] == labels[labels_neu].data)
                running_loss_neg += loss_neg.item() * inputs[labels_neg].size(0)
                running_corrects_neg += torch.sum(preds[labels_neg] == labels[labels_neg].data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss_neg = running_loss_neg / data_sizes[phase][0]
            epoch_acc_neg = running_corrects_neg / data_sizes[phase][0]
            epoch_loss_neu = running_loss_neu / data_sizes[phase][1]
            epoch_acc_neu = running_corrects_neu / data_sizes[phase][1]            
            epoch_loss_pos = running_loss_pos / data_sizes[phase][2]
            epoch_acc_pos = running_corrects_pos / data_sizes[phase][2]           
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('{} Pos_Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss_pos, epoch_acc_pos))
            print('{} Neu_Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss_neu, epoch_acc_neu))
            print('{} Neg_Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss_neg, epoch_acc_neg))
            
            if phase == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    time_elapsed = time.time() - since
    print('Training complete in {: .0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(params, lr = 0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

model = train_model(model_ft, criterion, optimizer_ft, 
                      exp_lr_scheduler, num_epochs=16)

torch.save(model, './TrainedModels/PretrainedDenseNet-FineTune-2208-3-lr001-Regularized-Corrected')
