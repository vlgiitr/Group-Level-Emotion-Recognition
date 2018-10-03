# IMPORTING MODULES
#----------------------------------------------------------------------------

from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle

import matplotlib.pyplot as plt


from sklearn.linear_model import SGDClassifier as SVM

#---------------------------------------------------------------------------
# IMPORTANT PARAMETERS
#---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else 'cpu'
root_dir = "../Dataset/"
epochs = 1
batch_size = 8
maxFaces = 15
label_to_name = { 0 : 'Negative',
                1 : 'Neutral',
                2 : 'Positive'}

#---------------------------------------------------------------------------
# DATASET AND LOADERS
#---------------------------------------------------------------------------

test_data_filelist = sorted(os.listdir('../Dataset/emotiw/test_shared/test/'))

for i in test_data_filelist:
    if i[0] != 't':
        test_data_filelist.remove(i)

dataset_sizes = [len(test_data_filelist)]
print(dataset_sizes)

test_global_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_faces_data_transform = transforms.Compose([
        transforms.Resize((96,112)),
        transforms.ToTensor()
    ])

class EmotiWDataset(Dataset):
    
    def __init__(self, filelist, root_dir, transformGlobal=transforms.ToTensor(), transformFaces = transforms.ToTensor()):
        """
        Args:
            filelist: List of names of image/feature files.
            root_dir: Dataset directory
            transform (callable, optional): Optional transformer to be applied
                                            on an image sample.
        """
        
        self.filelist = filelist
        self.root_dir = root_dir
        self.transformGlobal = transformGlobal
        self.transformFaces = transformFaces
            
    def __len__(self):
        return (len(self.filelist))
    
    def __getitem__(self, idx):
        filename = self.filelist[idx]
        filename = filename[:-4]
        #IMAGE 

        image = Image.open(self.root_dir+'emotiw/test_shared/test/'+filename+'.jpg')
        
        if self.transformGlobal:
            image = self.transformGlobal(image)
        if image.shape[0] == 1:
            image_1 = np.zeros((3, 224, 224), dtype = float)
            image_1[0] = image
            image_1[1] = image
            image_1[2] = image
            image = image_1
            image = torch.FloatTensor(image.tolist()) 
        
        #FEATURES FROM MTCNN

        features = np.load(self.root_dir+'FaceFeatures/test/'+filename+'.npz')['a']
        numberFaces = features.shape[0]
        
        maxNumber = min(numberFaces, maxFaces)
        

        features1 = np.zeros((maxFaces, 256), dtype = 'float32')
        
        for i in range(maxNumber):
            features1[i] = features[i]
        features1 = torch.from_numpy(features1)

        #ALIGNED CROPPED FACE IMAGES

        features2 = np.zeros((maxFaces, 3, 96, 112), dtype = 'float32')
#         print(maxNumber)
        
        for i in range(maxNumber):
            face = Image.open(self.root_dir + 'AlignedCroppedImages/test/' + filename+ '_' + str(i) + '.jpg')
                
            if self.transformFaces:
                face = self.transformFaces(face)
                
            features2[i] = face.numpy()
            
        features2 = torch.from_numpy(features2)
        
        labels = 1

        #SAMPLE
        sample = {'filename': filename, 'image': image, 'features_mtcnn': features1, 'features_aligned':features2, 'label':labels, 'numberFaces': numberFaces}
        return sample

test_dataset = EmotiWDataset(test_data_filelist, root_dir, transformGlobal = test_global_data_transform, transformFaces=test_faces_data_transform)

test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, num_workers = 0)
print('Dataset Loaded')

#---------------------------------------------------------------------------
# SPHEREFACE MODEL FOR ALIGNED MODELS
#---------------------------------------------------------------------------

class LSoftmaxLinear(nn.Module):

    def __init__(self, input_dim, output_dim, margin):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))

        self.divisor = math.pi / self.margin
        self.coeffs = binom(margin, range(0, margin + 1, 2))
        self.cos_exps = range(self.margin, -1, -2)
        self.sin_sq_exps = range(len(self.cos_exps))
        self.signs = [1]
        for i in range(1, len(self.sin_sq_exps)):
            self.signs.append(self.signs[-1] * -1)

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight.data.t())

    def find_k(self, cos):
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            logit = input.matmul(self.weight)
            batch_size = logit.size(0)
            logit_target = logit[range(batch_size), target]
            weight_target_norm = self.weight[:, target].norm(p=2, dim=0)
            input_norm = input.norm(p=2, dim=1)
            norm_target_prod = weight_target_norm * input_norm
            cos_target = logit_target / (norm_target_prod + 1e-10)
            sin_sq_target = 1 - cos_target**2
            
            weight_nontarget_norm = self.weight.norm(p=2, dim=0)
            
            norm_nontarget_prod = torch.zeros((batch_size,numClasses), dtype = torch.float)
            logit2 = torch.zeros((batch_size,numClasses), dtype = torch.float)
            logit3 = torch.zeros((batch_size,numClasses), dtype = torch.float)

            norm_nontarget_prod = norm_nontarget_prod.to(device)
            logit2 = logit2.to(device)
            logit3 = logit3.to(device)
            
            for i in range(numClasses):
                norm_nontarget_prod[:, i] = weight_nontarget_norm[i] * input_norm 
                logit2[:, i] = norm_target_prod / (norm_nontarget_prod[:, i] + 1e-10)
            
            for i in range(batch_size):
                for j in range(numClasses):
                    logit3[i][j] = logit2[i][j] * logit[i][j]

            num_ns = self.margin//2 + 1
            coeffs = Variable(input.data.new(self.coeffs))
            cos_exps = Variable(input.data.new(self.cos_exps))
            sin_sq_exps = Variable(input.data.new(self.sin_sq_exps))
            signs = Variable(input.data.new(self.signs))

            cos_terms = cos_target.unsqueeze(1) ** cos_exps.unsqueeze(0)
            sin_sq_terms = (sin_sq_target.unsqueeze(1)
                            ** sin_sq_exps.unsqueeze(0))

            cosm_terms = (signs.unsqueeze(0) * coeffs.unsqueeze(0)
                          * cos_terms * sin_sq_terms)
            cosm = cosm_terms.sum(1)
            k = self.find_k(cos_target)
            
            ls_target = norm_target_prod * (((-1)**k * cosm) - 2*k)
            logit3[range(batch_size), target] = ls_target
            return logit
        else:
            assert target is None
            return input.matmul(self.weight)

class sphere20a(nn.Module):
    def __init__(self,classnum=3,feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = LSoftmaxLinear(512,self.classnum, 4)

    def forward(self, x, y):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = (self.fc5(x))
#         print(x)
        if self.feature: return x

        x = self.fc6(x)
#         x = self.fc6(x, None)

        return x

#---------------------------------------------------------------------------
# MODEL 1
# Pretrained EmotiW DenseNet (DenseNet161_EmotiW)
#---------------------------------------------------------------------------

global_model = torch.load('../TrainedModels/TrainDataset/DenseNet161_EmotiW', map_location=lambda storage, loc: storage)
model1 = global_model
print('Pretrained EmotiW DenseNet Loaded! (Model 1)')

#---------------------------------------------------------------------------
# MODEL 2
# Pretrained EmotiC DenseNet (densenet_emotiw_pretrainemotic_lr001)
#---------------------------------------------------------------------------

model2 = models.densenet161(pretrained=False)
num_ftrs = model2.classifier.in_features
model2.classifier = nn.Linear(num_ftrs, 3)

model2 = model2.to(device)
model2 = nn.DataParallel(model2)
model2.load_state_dict(torch.load('../TrainedModels/TrainDataset/densenet_emotiw_pretrainemotic_lr001.pt', map_location=lambda storage, loc: storage))
model2 = model2.module

print('Pretrained EmotiC DenseNet Loaded! (Model 2)')

#---------------------------------------------------------------------------
# MODEL 3
# Aligned Model Global Level (AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.non_align_model = non_align_model
    
    def forward(self, face_features_initial, numberFaces, labels):
        
        maxNumber = np.minimum(numberFaces, maxFaces).float()
        maxNumber = maxNumber.to(device)

        face_features = torch.zeros((face_features_initial.shape[0],maxFaces,3), dtype = torch.float)
        
        for j in range(face_features_initial.shape[0]):
            face = face_features_initial[j]
            tensor = torch.zeros((2,), dtype=torch.long)
            faceLabels = tensor.new_full((maxFaces,), labels[j], dtype = torch.long)
            faceLabels = faceLabels.to(device)
            face_features[j, :, :] = self.non_align_model.forward(face, faceLabels)
            
        face_features = face_features.to(device)
        
        face_features_sum = torch.zeros((face_features_initial.shape[0], 3), dtype = torch.float)
        face_features_sum = face_features_sum.to(device)
        
        face_features_avg = torch.zeros((face_features_initial.shape[0], 3), dtype = torch.float)
        face_features_avg = face_features_avg.to(device)

        for i in range(face_features_initial.shape[0]):
            for j in range(int(maxNumber[i])):
                face_features_sum[i] = face_features_sum[i] + face_features[i][j]
                
            if int(maxNumber[i]) != 0:
                y = float(maxNumber[i])
                face_features_avg[i] = face_features_sum[i] / y

        return face_features_avg

aligned_model_global_level_path = "../TrainedModels/TrainDataset/AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax"
align_model = torch.load(aligned_model_global_level_path, map_location=lambda storage, loc: storage).module
model3 = align_model
print('Aligned Model Global Level Loaded! (Model 3)')

#---------------------------------------------------------------------------
# MODEL 4
# Aligned Model Image Level Trained (AlignedModel_EmotiW_lr01_Softmax)
#---------------------------------------------------------------------------

aligned_model_image_level_path = '../TrainedModels/TrainDataset/AlignedModel_EmotiW_lr01_Softmax'
align_model = torch.load(aligned_model_image_level_path, map_location=lambda storage, loc: storage).module

class FaceAttention(nn.Module):
    def __init__(self, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.non_align_model = non_align_model
    
    def forward(self, face_features_initial, numberFaces, labels):
        
        maxNumber = np.minimum(numberFaces, maxFaces).float()
        maxNumber = maxNumber.to(device)

        face_features = torch.zeros((face_features_initial.shape[0],maxFaces,3), dtype = torch.float)
        
        for j in range(face_features_initial.shape[0]):
            face = face_features_initial[j]
            tensor = torch.zeros((2,), dtype=torch.long)
            faceLabels = tensor.new_full((maxFaces,), labels[j], dtype = torch.long)
            faceLabels = faceLabels.to(device)
            face_features[j, :, :] = self.non_align_model.forward(face, faceLabels)
            
        face_features = face_features.to(device)
        
        face_features_sum = torch.zeros((face_features_initial.shape[0], 3), dtype = torch.float)
        face_features_sum = face_features_sum.to(device)
        
        face_features_avg = torch.zeros((face_features_initial.shape[0], 3), dtype = torch.float)
        face_features_avg = face_features_avg.to(device)

        for i in range(face_features_initial.shape[0]):
            for j in range(int(maxNumber[i])):
                face_features_sum[i] = face_features_sum[i] + face_features[i][j]
                
            if int(maxNumber[i]) != 0:
                y = float(maxNumber[i])
                face_features_avg[i] = face_features_sum[i] / y

        return face_features_avg


model4 = FaceAttention(align_model)
print('Aligned Model Image Level Loaded! (Model 4)')

#---------------------------------------------------------------------------
# MODEL 5
# Avg. Face Features Concat Model (PretrainedDenseNetAvgFaceFeatures-FineTune-2208-3-NoSoftmax-Reg-lr001)
#---------------------------------------------------------------------------

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
    
model5 = torch.load('../TrainedModels/TrainDataset/PretrainedDenseNet-FineTune-2208-3-lr001-Regularized-Corrected', map_location=lambda storage, loc: storage).module
print('Avg. Face Features Concat Model Loaded! (Model 5)')

#---------------------------------------------------------------------------
# MODEL 6
# Face Attention Model (EmotiC) using 3rd Para Attention 
# (FaceAttention_AlignedModel_FullTrain_3para_lr001_dropout_BN_SoftmaxLr01_EmotiC)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, global_model, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.global_model = global_model
        self.non_align_model = non_align_model
        
        self.global_fc3_debug = nn.Linear(320, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)

        self.global_fc = nn.Linear(256, 64)
        nn.init.kaiming_normal_(self.global_fc.weight)
        self.global_fc.bias.data.fill_(0.01)   

        self.global_fc_dropout = nn.Dropout(p = 0.5)
        self.global_fc_main_dropout = nn.Dropout(p = 0.5)
        self.non_align_model_dropout = nn.Dropout(p = 0.5)

        self.bn_debug_face = nn.BatchNorm1d(64, affine=False)
        self.bn_debug_global = nn.BatchNorm1d(256, affine=False)
    
    def forward(self, image, face_features_initial, numberFaces, labels):

        features = self.global_model.forward(image)

        global_features_main = self.global_fc_main_dropout(features)
        
        global_features = self.global_fc_dropout(self.global_fc(global_features_main))

        global_features = global_features.view(-1,1,64)

        batch_size = global_features.shape[0]
        
        maxNumber = np.minimum(numberFaces, maxFaces)

        face_features = torch.zeros((batch_size,maxFaces,64), dtype = torch.float)
        
        face_features = face_features.to(device)

        for j in range(batch_size):
            face = face_features_initial[j]
            face_features[j, :, :] = self.non_align_model.forward(face, labels)
        
        face_features = self.non_align_model_dropout(face_features)

        face_features = face_features.view(batch_size, 64, -1)

        mask = np.zeros((batch_size,1,maxFaces), dtype = 'float32')
        for j in range(batch_size):
            for i in range(maxFaces - (int(maxNumber[j]))):
                mask[j][0][int(numberFaces[j]) + i] = float('-inf')
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        attention_scores = torch.bmm(global_features, face_features) #(batch_size, 1, 256) x (batch_size, 256, nFaces) = (batch_size, 1, nFaces)
        attention_scores = attention_scores+mask
                
        #Convert Scores to Weight
        attention_scores = F.softmax(attention_scores, dim = -1)
        
        attention_weights = attention_scores
        
        attention_weights = Variable(attention_scores)
        
        for i in range(len(maxNumber)):
            if maxNumber[i] == 0:
                for j in range(maxFaces):
                    attention_weights[i][0][j] =  0 
        
        #Taking Weighted Average of Face Featrues
        face_features = face_features.view(batch_size, -1, 64) #(batch_size, nFaces, 256)
        attention_scores = attention_weights.view(batch_size, 1, -1) #(batch_size, 1, nFaces)
        attended_face_features = torch.bmm(attention_scores, face_features)
        
        #Concatenating Global and Attended Face Features 
        attended_face_features = attended_face_features.view(batch_size, -1)
        attended_face_features = self.bn_debug_face(attended_face_features)
        global_features_main = self.bn_debug_global(global_features_main)
        final_features = torch.cat((attended_face_features, global_features_main), dim=1)
        
        x = (self.global_fc3_debug(final_features))        
        return x

model6 = torch.load('../TrainedModels/TrainDataset/FaceAttention_AlignedModel_FullTrain_3para_lr001_dropout_BN_SoftmaxLr01_EmotiC', map_location=lambda storage, loc: storage).module
print('Face Attention Model (EmotiC) using 3rd Para Attention Loaded! (Model 6)')

#---------------------------------------------------------------------------
# MODEL 7
# Face Attention Model (EmotiC) using 4th Para Attention 
# (FaceAttention_AlignedModel_FullTrain_4para_lr001_dropout_BN_SoftmaxLr01_EmotiC)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, global_model, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.global_model = global_model
        self.non_align_model = non_align_model
        
        self.global_fc3_debug = nn.Linear(512, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)

        self.attentionfc1 = nn.Linear(256, 64)
        nn.init.kaiming_normal_(self.attentionfc1.weight)
        self.attentionfc1.bias.data.fill_(0.01)   

        self.attentionfc2 = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.attentionfc2.weight)
        self.attentionfc2.bias.data.fill_(0.01)

        self.attentionfc1_dropout = nn.Dropout(p = 0.5)
        self.global_fc_main_dropout = nn.Dropout(p = 0.5)
        self.non_align_model_dropout = nn.Dropout(p = 0.5)

        self.bn_debug_face = nn.BatchNorm1d(256, affine=False)
        self.bn_debug_global = nn.BatchNorm1d(256, affine=False)
    
    def forward(self, image, face_features_initial, numberFaces, labels):

        features = self.global_model.forward(image)

        global_features = self.global_fc_main_dropout(features)
        
        batch_size = global_features.shape[0]

        global_features = global_features.view(-1,1,256)

        
        maxNumber = np.minimum(numberFaces, maxFaces)

        face_features = torch.zeros((batch_size,maxFaces,256), dtype = torch.float)
        
        face_features = face_features.to(device)

        mid_face_features = torch.zeros((batch_size, maxFaces, 1), dtype = torch.float)
        face_features_inter = torch.zeros((batch_size, maxFaces, 64), dtype = torch.float)
        face_features_inter = face_features_inter.to(device)
        mid_face_features = mid_face_features.to(device)

        for j in range(batch_size):
            face = face_features_initial[j]
            face_features[j, :, :] = self.non_align_model_dropout(self.non_align_model.forward(face, labels))
            face_features_inter[j] = self.attentionfc1_dropout(self.attentionfc1(face_features[j]))
            mid_face_features[j] = self.attentionfc2(face_features_inter[j])
        
    
        mid_face_features = mid_face_features.view(batch_size, 1, maxFaces)

        mask = np.zeros((batch_size,1,maxFaces), dtype = 'float32')
        for j in range(batch_size):
            for i in range(maxFaces - (int(maxNumber[j]))):
                mask[j][0][int(numberFaces[j]) + i] = float('-inf')
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        attention_scores = mid_face_features + mask
        
        #Convert Scores to Weight
        attention_scores = F.softmax(attention_scores, dim = -1)
        
        attention_weights = Variable(attention_scores)
        
        for i in range(len(maxNumber)):
            if maxNumber[i] == 0:
                for j in range(maxFaces):
                    attention_weights[i][0][j] =  0 
        
        #Taking Weighted Average of Face Featrues
        face_features = face_features.view(batch_size, -1, 256) #(batch_size, nFaces, 256)
        attention_scores = attention_weights.view(batch_size, 1, -1) #(batch_size, 1, nFaces)
        attended_face_features = torch.bmm(attention_scores, face_features)
        
        #Concatenating Global and Attended Face Features 
        attended_face_features = attended_face_features.view(batch_size, -1)
        global_features = global_features.view(batch_size, -1)
        
        attended_face_features = self.bn_debug_face(attended_face_features)
        global_features = self.bn_debug_global(global_features)

        final_features = torch.cat((attended_face_features, global_features), dim=1)
        
        x = (self.global_fc3_debug(final_features))        
        return x
    
model7 = torch.load('../TrainedModels/TrainDataset/FaceAttention_AlignedModel_FullTrain_4para_lr001_dropout_BN_SoftmaxLr01_EmotiC', map_location=lambda storage, loc: storage).module
print('Face Attention Model (EmotiC) using 4rd Para Attention Loaded! (Model 7)')

#---------------------------------------------------------------------------
# MODEL 8
# Face Attention Model using 4th Para Attention (FaceAttention_AlignedModel_FullTrain_4para_lr01_dropout_BN_SoftmaxLr01)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, global_model, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.global_model = global_model
        self.non_align_model = non_align_model
        
        self.global_fc3_debug = nn.Linear(512, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)

        self.attentionfc1 = nn.Linear(256, 64)
        nn.init.kaiming_normal_(self.attentionfc1.weight)
        self.attentionfc1.bias.data.fill_(0.01)   

        self.attentionfc2 = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.attentionfc2.weight)
        self.attentionfc2.bias.data.fill_(0.01)

        self.global_fc_main = nn.Linear(2208, 256)
        nn.init.kaiming_normal_(self.global_fc_main.weight)
        self.global_fc_main.bias.data.fill_(0.01)

        self.attentionfc1_dropout = nn.Dropout(p = 0.5)
        self.global_fc_main_dropout = nn.Dropout(p = 0.5)
        self.non_align_model_dropout = nn.Dropout(p = 0.5)

        self.bn_debug_face = nn.BatchNorm1d(256, affine=False)
        self.bn_debug_global = nn.BatchNorm1d(256, affine=False)
    
    def forward(self, image, face_features_initial, numberFaces, labels):

        features = self.global_model.forward(image)

        out = F.relu(features, inplace = False)
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        global_features_initial = Variable(global_features_initial)

        global_features_initial = global_features_initial.view(-1,2208)

        global_features = self.global_fc_main_dropout(self.global_fc_main(global_features_initial))
        
        batch_size = global_features.shape[0]

        global_features = global_features.view(-1,1,256)

        
        maxNumber = np.minimum(numberFaces, maxFaces)

        face_features = torch.zeros((batch_size,maxFaces,256), dtype = torch.float)
        
        face_features = face_features.to(device)

        mid_face_features = torch.zeros((batch_size, maxFaces, 1), dtype = torch.float)
        face_features_inter = torch.zeros((batch_size, maxFaces, 64), dtype = torch.float)
        face_features_inter = face_features_inter.to(device)
        mid_face_features = mid_face_features.to(device)

        for j in range(batch_size):
            face = face_features_initial[j]
            face_features[j, :, :] = self.non_align_model_dropout(self.non_align_model.forward(face, labels))
            face_features_inter[j] = self.attentionfc1_dropout(self.attentionfc1(face_features[j]))
            mid_face_features[j] = self.attentionfc2(face_features_inter[j])
            
        mid_face_features = mid_face_features.view(batch_size, 1, maxFaces)

        mask = np.zeros((batch_size,1,maxFaces), dtype = 'float32')
        for j in range(batch_size):
            for i in range(maxFaces - (int(maxNumber[j]))):
                mask[j][0][int(numberFaces[j]) + i] = float('-inf')
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        attention_scores = mid_face_features + mask
        
        #Convert Scores to Weight
        attention_scores = F.softmax(attention_scores, dim = -1)
        
        attention_weights = Variable(attention_scores)
        
        for i in range(len(maxNumber)):
            if maxNumber[i] == 0:
                for j in range(maxFaces):
                    attention_weights[i][0][j] =  0 
        
        #Taking Weighted Average of Face Featrues
        face_features = face_features.view(batch_size, -1, 256) #(batch_size, nFaces, 256)
        attention_scores = attention_weights.view(batch_size, 1, -1) #(batch_size, 1, nFaces)
        attended_face_features = torch.bmm(attention_scores, face_features)
        
        #Concatenating Global and Attended Face Features 
        attended_face_features = attended_face_features.view(batch_size, -1)
        global_features = global_features.view(batch_size, -1)
        
        attended_face_features = self.bn_debug_face(attended_face_features)
        global_features = self.bn_debug_global(global_features)

        final_features = torch.cat((attended_face_features, global_features), dim=1)
        
        x = (self.global_fc3_debug(final_features))        
        return x
    
model8 = torch.load('../TrainedModels/TrainDataset/FaceAttention_AlignedModel_FullTrain_4para_lr01_dropout_BN_SoftmaxLr01', map_location=lambda storage, loc: storage).module
print('Face Attention Model using 4rd Para Attention Loaded! (Model 8)')

#---------------------------------------------------------------------------
# MODEL 9
# Aligned Model Global Level (AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax_br128)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.non_align_model = non_align_model
    
    def forward(self, face_features_initial, numberFaces, labels):
        
        maxNumber = np.minimum(numberFaces, maxFaces).float()
        maxNumber = maxNumber.to(device)

        face_features = torch.zeros((face_features_initial.shape[0],maxFaces,3), dtype = torch.float)
        
        for j in range(face_features_initial.shape[0]):
            face = face_features_initial[j]
            tensor = torch.zeros((2,), dtype=torch.long)
            faceLabels = tensor.new_full((maxFaces,), labels[j], dtype = torch.long)
            faceLabels = faceLabels.to(device)
            face_features[j, :, :] = self.non_align_model.forward(face, faceLabels)
            
        face_features = face_features.to(device)
        
        face_features_sum = torch.zeros((face_features_initial.shape[0], 3), dtype = torch.float)
        face_features_sum = face_features_sum.to(device)
        
        face_features_avg = torch.zeros((face_features_initial.shape[0], 3), dtype = torch.float)
        face_features_avg = face_features_avg.to(device)

        for i in range(face_features_initial.shape[0]):
            for j in range(int(maxNumber[i])):
                face_features_sum[i] = face_features_sum[i] + face_features[i][j]
                
            if int(maxNumber[i]) != 0:
                y = float(maxNumber[i])
                face_features_avg[i] = face_features_sum[i] / y

        return face_features_avg

aligned_model_global_level_path = "../TrainedModels/TrainDataset/AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax_br128"
align_model = torch.load(aligned_model_global_level_path, map_location=lambda storage, loc: storage).module
model9 = align_model
print('Aligned Model Global Level Loaded! (Model 9)')

#---------------------------------------------------------------------------
# MODEL 10
# Aligned Model Global Level Trained (AlignedModelTrainerLSoftmax_AlignedModel_EmotiW_lr001)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.non_align_model = non_align_model
    
    def forward(self, face_features_initial, numberFaces, labels, phase):
        
        maxNumber = np.minimum(numberFaces, maxFaces).float()
        maxNumber = maxNumber.to(device)

        face_features = torch.zeros((face_features_initial.shape[0],maxFaces,3), dtype = torch.float)
        
        for j in range(face_features_initial.shape[0]):
            face = face_features_initial[j]
            tensor = torch.zeros((2,), dtype=torch.long)
            faceLabels = tensor.new_full((maxFaces,), labels[j], dtype = torch.long)
            faceLabels = faceLabels.to(device)

            if phase == 0:
                face_features[j, :, :] = self.non_align_model.forward(face, faceLabels)
            else:
                face_features[j, :, :] = self.non_align_model.forward(face, None)
            
        face_features = face_features.to(device)
        face_features_sum = torch.zeros((face_features_initial.shape[0], 3), dtype = torch.float)

        face_features_sum = face_features_sum.to(device)
        
        face_features_avg = torch.zeros((face_features_initial.shape[0], 3), dtype = torch.float)

        face_features_avg = face_features_avg.to(device)

        for i in range(face_features_initial.shape[0]):
            for j in range(int(maxNumber[i])):
                face_features_sum[i] = face_features_sum[i] + face_features[i][j]
                
            if int(maxNumber[i]) != 0:
                y = float(maxNumber[i])
                face_features_avg[i] = face_features_sum[i] / y

        return face_features_avg

aligned_model_global_level_path = "../TrainedModels/TrainDataset/AlignedModelTrainerLSoftmax_AlignedModel_EmotiW_lr001"
align_model = torch.load(aligned_model_global_level_path, map_location=lambda storage, loc: storage).module
model10 = align_model
print('Aligned L-softmax Model Global Level Loaded! (Model 10)')

#---------------------------------------------------------------------------
# MODEL 11
# FaceAttention Similarity Attention Mechanism (FaceAttention_AlignedModel_FullTrain_lr001_dropout_BN_SoftmaxLr01)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, global_model, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.global_model = global_model
        self.non_align_model = non_align_model
        
        self.global_fc_main = nn.Linear(2208, 256)
        nn.init.kaiming_normal_(self.global_fc_main.weight)
        self.global_fc_main.bias.data.fill_(0.01)

        self.global_fc3_debug = nn.Linear(512, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)   

        self.global_fc_main_dropout = nn.Dropout(p = 0.5)
        self.non_align_model_dropout = nn.Dropout(p = 0.5)

        self.bn_debug_face = nn.BatchNorm1d(256, affine=False)
        self.bn_debug_global = nn.BatchNorm1d(256, affine=False)
    
    def forward(self, image, face_features_initial, numberFaces, labels):

        features = self.global_model.forward(image)

        out = F.relu(features, inplace = False)
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        global_features_initial = Variable(global_features_initial)

        global_features_initial = global_features_initial.view(-1,2208)

        global_features = self.global_fc_main_dropout(self.global_fc_main(global_features_initial))
        
        global_features = global_features.view(-1,1,256)

        batch_size = global_features.shape[0]
        
        maxNumber = np.minimum(numberFaces, maxFaces)

        face_features = torch.zeros((batch_size,maxFaces,256), dtype = torch.float)
        
        face_features = face_features.to(device)

        for j in range(batch_size):
            face = face_features_initial[j]
            face_features[j, :, :] = self.non_align_model.forward(face, labels)
        
        face_features = self.non_align_model_dropout(face_features)

        face_features = face_features.view(batch_size, 256, -1)

        mask = np.zeros((batch_size,1,maxFaces), dtype = 'float32')
        for j in range(batch_size):
            for i in range(maxFaces - (int(maxNumber[j]))):
                mask[j][0][int(numberFaces[j]) + i] = float('-inf')
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        attention_scores = torch.bmm(global_features, face_features) #(batch_size, 1, 256) x (batch_size, 256, nFaces) = (batch_size, 1, nFaces)
        attention_scores = attention_scores+mask
        
        
        #Convert Scores to Weight
        attention_scores = F.softmax(attention_scores, dim = -1)
        
        attention_weights = attention_scores

        attention_weights = Variable(attention_scores)
        
        for i in range(len(maxNumber)):
            if maxNumber[i] == 0:
                for j in range(maxFaces):
                    attention_weights[i][0][j] =  0 
        
        #Taking Weighted Average of Face Featrues
        face_features = face_features.view(batch_size, -1, 256) #(batch_size, nFaces, 256)
        attention_scores = attention_weights.view(batch_size, 1, -1) #(batch_size, 1, nFaces)
        attended_face_features = torch.bmm(attention_scores, face_features)
        
        #Concatenating Global and Attended Face Features 
        attended_face_features = attended_face_features.view(batch_size, -1)
        global_features = global_features.view(batch_size, -1)
        attended_face_features = self.bn_debug_face(attended_face_features)
        global_features = self.bn_debug_global(global_features)
        final_features = torch.cat((attended_face_features, global_features), dim=1)
        
        x = (self.global_fc3_debug(final_features))        
        return x
    
model11 = torch.load('../TrainedModels/TrainDataset/FaceAttention_AlignedModel_FullTrain_lr001_dropout_BN_SoftmaxLr01', map_location=lambda storage, loc: storage).module
print('FaceAttention Similarity Attention Mechanism Model Loaded! (Model 11)')

#---------------------------------------------------------------------------
# MODEL 12
# Face Attention Model using 3rd Para Attention 
# (FaceAttention_AlignedModel_FullTrain_3para_lr001_dropout_BN_SoftmaxLr01)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, global_model, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.global_model = global_model
        self.non_align_model = non_align_model
        
        self.global_fc3_debug = nn.Linear(320, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)

        self.global_fc = nn.Linear(256, 64)
        nn.init.kaiming_normal_(self.global_fc.weight)
        self.global_fc.bias.data.fill_(0.01)   

        self.global_fc_dropout = nn.Dropout(p = 0.5)
        self.global_fc_main_dropout = nn.Dropout(p = 0.5)
        self.non_align_model_dropout = nn.Dropout(p = 0.5)

        self.bn_debug_face = nn.BatchNorm1d(64, affine=False)
        self.bn_debug_global = nn.BatchNorm1d(256, affine=False)
        
        self.global_fc_main = nn.Linear(2208, 256)
        nn.init.kaiming_normal_(self.global_fc_main.weight)
        self.global_fc_main.bias.data.fill_(0.01)
    
    def forward(self, image, face_features_initial, numberFaces, labels):

        features = self.global_model.forward(image)

        out = F.relu(features, inplace = False)
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        global_features_initial = Variable(global_features_initial)

        global_features_initial = global_features_initial.view(-1,2208)

        global_features_main = self.global_fc_main_dropout(self.global_fc_main(global_features_initial))
        
        global_features = self.global_fc_dropout(self.global_fc(global_features_main))

        global_features = global_features.view(-1,1,64)

        batch_size = global_features.shape[0]
        
        maxNumber = np.minimum(numberFaces, maxFaces)

        face_features = torch.zeros((batch_size,maxFaces,64), dtype = torch.float)
        
        face_features = face_features.to(device)

        for j in range(batch_size):
            face = face_features_initial[j]
            face_features[j, :, :] = self.non_align_model.forward(face, labels)
        
        face_features = self.non_align_model_dropout(face_features)

        face_features = face_features.view(batch_size, 64, -1)

        mask = np.zeros((batch_size,1,maxFaces), dtype = 'float32')
        for j in range(batch_size):
            for i in range(maxFaces - (int(maxNumber[j]))):
                mask[j][0][int(numberFaces[j]) + i] = float('-inf')
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        attention_scores = torch.bmm(global_features, face_features) #(batch_size, 1, 256) x (batch_size, 256, nFaces) = (batch_size, 1, nFaces)
        attention_scores = attention_scores+mask
                
        #Convert Scores to Weight
        attention_scores = F.softmax(attention_scores, dim = -1)
        
        attention_weights = attention_scores
        
        attention_weights = Variable(attention_scores)
        
        for i in range(len(maxNumber)):
            if maxNumber[i] == 0:
                for j in range(maxFaces):
                    attention_weights[i][0][j] =  0 
        
        #Taking Weighted Average of Face Featrues
        face_features = face_features.view(batch_size, -1, 64) #(batch_size, nFaces, 256)
        attention_scores = attention_weights.view(batch_size, 1, -1) #(batch_size, 1, nFaces)
        attended_face_features = torch.bmm(attention_scores, face_features)
        
        #Concatenating Global and Attended Face Features 
        attended_face_features = attended_face_features.view(batch_size, -1)
        attended_face_features = self.bn_debug_face(attended_face_features)
        global_features_main = self.bn_debug_global(global_features_main)
        final_features = torch.cat((attended_face_features, global_features_main), dim=1)
        
        x = (self.global_fc3_debug(final_features))        
        return x

model12 = torch.load('../TrainedModels/TrainDataset/FaceAttention_AlignedModel_FullTrain_3para_lr001_dropout_BN_SoftmaxLr01', map_location=lambda storage, loc: storage).module
print('Face Attention Model using 3rd Para Attention Loaded! (Model 12)')

#---------------------------------------------------------------------------
# MODEL 13
# Face Attention Model (EmotiC) using 4th Para Attention 
# (FaceAttention_AlignedModel_FullTrain_4para_lr01_dropout_BN_SoftmaxLr01_EmotiC)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, global_model, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.global_model = global_model
        self.non_align_model = non_align_model
        
        self.global_fc3_debug = nn.Linear(512, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)

        self.attentionfc1 = nn.Linear(256, 64)
        nn.init.kaiming_normal_(self.attentionfc1.weight)
        self.attentionfc1.bias.data.fill_(0.01)   

        self.attentionfc2 = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.attentionfc2.weight)
        self.attentionfc2.bias.data.fill_(0.01)

        self.attentionfc1_dropout = nn.Dropout(p = 0.5)
        self.global_fc_main_dropout = nn.Dropout(p = 0.5)
        self.non_align_model_dropout = nn.Dropout(p = 0.5)

        self.bn_debug_face = nn.BatchNorm1d(256, affine=False)
        self.bn_debug_global = nn.BatchNorm1d(256, affine=False)
    
    def forward(self, image, face_features_initial, numberFaces, labels):

        features = self.global_model.forward(image)

        global_features = self.global_fc_main_dropout(features)
        
        batch_size = global_features.shape[0]

        global_features = global_features.view(-1,1,256)

        
        maxNumber = np.minimum(numberFaces, maxFaces)

        face_features = torch.zeros((batch_size,maxFaces,256), dtype = torch.float)
        
        face_features = face_features.to(device)

        mid_face_features = torch.zeros((batch_size, maxFaces, 1), dtype = torch.float)
        face_features_inter = torch.zeros((batch_size, maxFaces, 64), dtype = torch.float)
        face_features_inter = face_features_inter.to(device)
        mid_face_features = mid_face_features.to(device)

        for j in range(batch_size):
            face = face_features_initial[j]
            face_features[j, :, :] = self.non_align_model_dropout(self.non_align_model.forward(face, labels))
            face_features_inter[j] = self.attentionfc1_dropout(self.attentionfc1(face_features[j]))
            mid_face_features[j] = self.attentionfc2(face_features_inter[j])
        
    
        mid_face_features = mid_face_features.view(batch_size, 1, maxFaces)

        mask = np.zeros((batch_size,1,maxFaces), dtype = 'float32')
        for j in range(batch_size):
            for i in range(maxFaces - (int(maxNumber[j]))):
                mask[j][0][int(numberFaces[j]) + i] = float('-inf')
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        attention_scores = mid_face_features + mask
        
        #Convert Scores to Weight
        attention_scores = F.softmax(attention_scores, dim = -1)
        
        attention_weights = Variable(attention_scores)
        
        for i in range(len(maxNumber)):
            if maxNumber[i] == 0:
                for j in range(maxFaces):
                    attention_weights[i][0][j] =  0 
        
        #Taking Weighted Average of Face Featrues
        face_features = face_features.view(batch_size, -1, 256) #(batch_size, nFaces, 256)
        attention_scores = attention_weights.view(batch_size, 1, -1) #(batch_size, 1, nFaces)
        attended_face_features = torch.bmm(attention_scores, face_features)
        
        #Concatenating Global and Attended Face Features 
        attended_face_features = attended_face_features.view(batch_size, -1)
        global_features = global_features.view(batch_size, -1)
        
        attended_face_features = self.bn_debug_face(attended_face_features)
        global_features = self.bn_debug_global(global_features)

        final_features = torch.cat((attended_face_features, global_features), dim=1)
        
        x = (self.global_fc3_debug(final_features))        
        return x
    
model13 = torch.load('../TrainedModels/TrainDataset/FaceAttention_AlignedModel_FullTrain_4para_lr01_dropout_BN_SoftmaxLr01_EmotiC', map_location=lambda storage, loc: storage).module
print('Face Attention Model (EmotiC) using 4rd Para Attention Loaded! (Model 13)')

#---------------------------------------------------------------------------
# MODEL 14
# Face Attention Model using 4th Para Attention (FaceAttention_AlignedModel_FullTrain_4para_adam_dropout_BN_SoftmaxLr01)
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, global_model, non_align_model):
        super(FaceAttention, self).__init__()
        
        self.global_model = global_model
        self.non_align_model = non_align_model
        
        self.global_fc3_debug = nn.Linear(512, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)

        self.attentionfc1 = nn.Linear(256, 64)
        nn.init.kaiming_normal_(self.attentionfc1.weight)
        self.attentionfc1.bias.data.fill_(0.01)   

        self.attentionfc2 = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.attentionfc2.weight)
        self.attentionfc2.bias.data.fill_(0.01)

        self.global_fc_main = nn.Linear(2208, 256)
        nn.init.kaiming_normal_(self.global_fc_main.weight)
        self.global_fc_main.bias.data.fill_(0.01)

        self.attentionfc1_dropout = nn.Dropout(p = 0.5)
        self.global_fc_main_dropout = nn.Dropout(p = 0.5)
        self.non_align_model_dropout = nn.Dropout(p = 0.5)

        self.bn_debug_face = nn.BatchNorm1d(256, affine=False)
        self.bn_debug_global = nn.BatchNorm1d(256, affine=False)
    
    def forward(self, image, face_features_initial, numberFaces, labels):

        features = self.global_model.forward(image)

        out = F.relu(features, inplace = False)
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        global_features_initial = Variable(global_features_initial)

        global_features_initial = global_features_initial.view(-1,2208)

        global_features = self.global_fc_main_dropout(self.global_fc_main(global_features_initial))
        
        batch_size = global_features.shape[0]

        global_features = global_features.view(-1,1,256)

        
        maxNumber = np.minimum(numberFaces, maxFaces)

        face_features = torch.zeros((batch_size,maxFaces,256), dtype = torch.float)
        
        face_features = face_features.to(device)

        mid_face_features = torch.zeros((batch_size, maxFaces, 1), dtype = torch.float)
        face_features_inter = torch.zeros((batch_size, maxFaces, 64), dtype = torch.float)
        face_features_inter = face_features_inter.to(device)
        mid_face_features = mid_face_features.to(device)

        for j in range(batch_size):
            face = face_features_initial[j]
            face_features[j, :, :] = self.non_align_model_dropout(self.non_align_model.forward(face, labels))
            face_features_inter[j] = self.attentionfc1_dropout(self.attentionfc1(face_features[j]))
            mid_face_features[j] = self.attentionfc2(face_features_inter[j])
        
    
        mid_face_features = mid_face_features.view(batch_size, 1, maxFaces)

        mask = np.zeros((batch_size,1,maxFaces), dtype = 'float32')
        for j in range(batch_size):
            for i in range(maxFaces - (int(maxNumber[j]))):
                mask[j][0][int(numberFaces[j]) + i] = float('-inf')
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        attention_scores = mid_face_features + mask
        
        #Convert Scores to Weight
        attention_scores = F.softmax(attention_scores, dim = -1)
        
        attention_weights = Variable(attention_scores)
        
        for i in range(len(maxNumber)):
            if maxNumber[i] == 0:
                for j in range(maxFaces):
                    attention_weights[i][0][j] =  0 
        
        #Taking Weighted Average of Face Featrues
        face_features = face_features.view(batch_size, -1, 256) #(batch_size, nFaces, 256)
        attention_scores = attention_weights.view(batch_size, 1, -1) #(batch_size, 1, nFaces)
        attended_face_features = torch.bmm(attention_scores, face_features)
        
        #Concatenating Global and Attended Face Features 
        attended_face_features = attended_face_features.view(batch_size, -1)
        global_features = global_features.view(batch_size, -1)
        
        attended_face_features = self.bn_debug_face(attended_face_features)
        global_features = self.bn_debug_global(global_features)

        final_features = torch.cat((attended_face_features, global_features), dim=1)
        
        x = (self.global_fc3_debug(final_features))        
        return x
    
model14 = torch.load('../TrainedModels/TrainDataset/FaceAttention_AlignedModel_FullTrain_4para_adam_dropout_BN_SoftmaxLr01', map_location=lambda storage, loc: storage).module
print('Face Attention Model using 4rd Para Attention Loaded! (Model 14)')


#---------------------------------------------------------------------------
# ENSEMBLE
#---------------------------------------------------------------------------

class Ensemble(nn.Module):
    def __init__(self, model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11, model_12, model_13, model_14):
        super(Ensemble, self).__init__()
        
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self.model_4 = model_4
        self.model_5 = model_5
        self.model_6 = model_6
        self.model_7 = model_7
        self.model_8 = model_8
        self.model_9 = model_9
        self.model_10 = model_10
        self.model_11 = model_11
        self.model_12 = model_12
        self.model_13 = model_13
        self.model_14 = model_14

    def forward(self, image, labels, face_features_mtcnn, face_features_aligned, numberFaces, phase):
        
        output1 = self.model_1(image)
        output2 = self.model_2(image)
        output3 = self.model_3(face_features_aligned, numberFaces, labels)
        output4 = self.model_4(face_features_aligned, numberFaces, labels)
        output5 = self.model_5(image, face_features_mtcnn, numberFaces)
        output6 = self.model_6(image, face_features_aligned, numberFaces, labels)
        output7 = self.model_7(image, face_features_aligned, numberFaces, labels)
        output8 = self.model_8(image, face_features_aligned, numberFaces, labels)
        output9 = self.model_9(face_features_aligned, numberFaces, labels)
        output10 = self.model_10(face_features_aligned, numberFaces, labels, phase)
        output11 = self.model_11(image, face_features_aligned, numberFaces, labels)
        output12 = self.model_12(image, face_features_aligned, numberFaces, labels)
        output13 = self.model_13(image, face_features_aligned, numberFaces, labels)
        output14 = self.model_14(image, face_features_aligned, numberFaces, labels)
        
        output = 0 * output1 + 5 * output2 + 10 * output3 + 10 * output4 + 1 * output5 +  5 * output6 + 2 * output7 + 5 * output8
        return output, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, output11, output12, output13, output14

model_ft = Ensemble(model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14)
model_ft = model_ft.to(device)
model_ft = nn.DataParallel(model_ft)
print("Ensemble Loaded.")

#---------------------------------------------------------------------------
# TRAINING
#---------------------------------------------------------------------------

output_test_model1 = []
output_test_model2 = []
output_test_model3 = []
output_test_model4 = []
output_test_model5 = []
output_test_model6 = []
output_test_model7 = []
output_test_model8 = []
output_test_model9 = []
output_test_model10 = []
output_test_model11 = []
output_test_model12 = []
output_test_model13 = []
output_test_model14 = []
output_test = []
filename_test = []

def train_model(model, criterion = None, optimizer=None, scheduler=None, num_epochs = 1):
    
    since = time.time()
    
    for epoch in range(1):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)
        
        phase = 1
        
        dataloaders = test_dataloader
        model.eval()

        for i_batch, sample_batched in enumerate(dataloaders):
            filenames = sample_batched['filename']
            inputs = sample_batched['image']
            labels = sample_batched['label']
            face_features_mtcnn = sample_batched['features_mtcnn']
            face_features_aligned = sample_batched['features_aligned']
            numberFaces = sample_batched['numberFaces']
            inputs = inputs.to(device)
            labels = labels.to(device)
            face_features_mtcnn= face_features_mtcnn.to(device)
            face_features_aligned = face_features_aligned.to(device)
            numberFaces = numberFaces.to(device)

            with torch.set_grad_enabled(phase == 0):
                outputs, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, output11, output12, output13, output14 = model(inputs, labels, face_features_mtcnn, face_features_aligned, numberFaces, phase)  

                output_test.extend(outputs.data.cpu().numpy())
                output_test_model1.extend(output1.data.cpu().numpy())
                output_test_model2.extend(output2.data.cpu().numpy())
                output_test_model3.extend(output3.data.cpu().numpy())
                output_test_model4.extend(output4.data.cpu().numpy())
                output_test_model5.extend(output5.data.cpu().numpy())
                output_test_model6.extend(output6.data.cpu().numpy())
                output_test_model7.extend(output7.data.cpu().numpy())
                output_test_model8.extend(output8.data.cpu().numpy())
                output_test_model9.extend(output9.data.cpu().numpy())
                output_test_model10.extend(output10.data.cpu().numpy())
                output_test_model11.extend(output11.data.cpu().numpy())
                output_test_model12.extend(output12.data.cpu().numpy())
                output_test_model13.extend(output13.data.cpu().numpy())
                output_test_model14.extend(output14.data.cpu().numpy())

                filename_test.extend(filenames)    

    time_elapsed = time.time() - since
    return model

model = train_model(model_ft, None, None, None, num_epochs=epochs)

np.savez('test_data_fourteen_models_outputs', 
output_test = output_test,
output_test_model1 = output_test_model1,
output_test_model2 = output_test_model2,
output_test_model3 = output_test_model3,
output_test_model4 = output_test_model4,
output_test_model5 = output_test_model5,
output_test_model6 = output_test_model6,
output_test_model7 = output_test_model7,
output_test_model8 = output_test_model8, 
output_test_model9 = output_test_model9, 
output_test_model10 = output_test_model10, 
output_test_model11 = output_test_model11, 
output_test_model12 = output_test_model12, 
output_test_model13 = output_test_model13, 
output_test_model14 = output_test_model14,  
filename_test = filename_test)