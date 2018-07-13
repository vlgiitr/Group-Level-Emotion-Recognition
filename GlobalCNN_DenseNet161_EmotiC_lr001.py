#----------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from skimage import io, transform

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision import transforms, utils

#---------------------------------------------------------------------------
# IMPORTANT PARAMETERS
#---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else 'cpu'
root_dir = "Dataset/"
epochs = 15
batch_size = 32
maxFaces = 15
numClasses = 3

#---------------------------------------------------------------------------
# DATASET AND LOADERS
#---------------------------------------------------------------------------

class EmotiC(Dataset):
    """EmotiC dataset."""

    def __init__(self, annotations_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(annotations_file)
        self.labels = self.data['valence']
        self.folders = self.data['folder']
        self.images = self.data['image']
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.folders[idx], self.images[idx])
        print(idx)
        image = Image.open(img_name)
        image = np.asarray(image)
        
        if len(image) == 2:
            image = image[0]
            
        if len(image.shape) == 2:
            h = image.shape[0]
            w = image.shape[1]
            image_1 = np.zeros((h,w,3))
            for i in range(h):
                for j in range(w):
                    image_1[i][j][0] = image[i][j]
                    image_1[i][j][1] = image[i][j]
                    image_1[i][j][2] = image[i][j]
            image = image_1
        
        if image.shape[2] == 4:
            image = image[:,:,0:3]
            
        image = image / 255.0
        image = image.astype('float32')

        label = int(self.labels[idx]) - 1

        if label < 4:
        	label = 0
        elif label >=4 and label < 7:
        	label = 1
        elif label >= 6 and label < 10:
        	label = 2
        
        sample = {'image': image, 'label': label}


        if self.transform:
            sample = self.transform(sample)


        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        sample = {'image': img, 'label': label}
        
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        if new_h == h and new_w == w:
            top = 0
            left = 0
        elif new_h == h:
            top = 0
            left = np.random.randint(0, w-new_w)
        elif new_w == w:
            left = 0
            top = np.random.randint(0, h-new_h)
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = torch.LongTensor([label])
        return {'image': torch.FloatTensor(image.tolist()),
                'label': label}

data_transforms = transforms.Compose([
        Rescale(224),
        RandomCrop(224),
        ToTensor()
    ])

face_dataset_tr = EmotiC(annotations_file='train_annotations.npz',
                            root_dir='emotic')

face_dataset_va = EmotiC(annotations_file='val_annotations.npz',
                            root_dir='emotic')

face_dataset_train = EmotiC(annotations_file='train_annotations.npz',
                            root_dir='emotic', transform = data_transforms)

face_dataset_valid = EmotiC(annotations_file='val_annotations.npz',
                            root_dir='emotic', transform = data_transforms)


dataloaders_train = torch.utils.data.DataLoader(face_dataset_train,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=0)

dataloaders_valid = torch.utils.data.DataLoader(face_dataset_valid,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=0)

dataset_sizes = [len(face_dataset_train), len(face_dataset_valid)]
print(dataset_sizes)

#---------------------------------------------------------------------------
# MODEL DEFINITION
#---------------------------------------------------------------------------

model_ft = models.densenet161(pretrained=True)
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, 3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)
model_ft = torch.nn.DataParallel(model_ft)

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
                dataloaders = dataloaders_train
                scheduler.step()
                model.train()
            else:
                dataloaders = dataloaders_valid
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for i_batch, sample_batched in enumerate(dataloaders):
                inputs = sample_batched['image']
                labels = sample_batched['label']
                labels = labels.squeeze(1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 0):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 0:
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
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

optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, 
                      exp_lr_scheduler, num_epochs=epochs)

torch.save(model_ft.state_dict(), "../TrainedModels/densenet_emotic_lr001.pt")