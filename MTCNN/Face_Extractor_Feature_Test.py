#---------------------------------------------------------------------------
# IMPORT MODULES
#---------------------------------------------------------------------------

from PIL import Image

import torch
from torchvision import transforms, datasets
import numpy as np
import os
import numpy as np
import torch
from torch.autograd import Variable
from src.get_nets import PNet, RNet, ONet
from src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from src.first_stage import run_first_stage
import torch.nn as nn

#---------------------------------------------------------------------------
# PATH DEFINITIONS
#---------------------------------------------------------------------------

processed_dataset_path = '../Dataset/FaceFeatures/test/'

#---------------------------------------------------------------------------
# MTCNN MODEL DEFINITION for EXTRACTING FACE FEATURES
#---------------------------------------------------------------------------

pnet = PNet()
rnet = RNet()
onet = ONet()
onet.eval()

class OnetFeatures(nn.Module):
    def __init__(self, original_model):
        super(OnetFeatures, self).__init__()
        self.features = nn.Sequential(*list(onet.children())[:-3])
        
    def forward(self, x):
        x = self.features(x)
        return x

def get_face_features(image, min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """

    # LOAD MODELS
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0: 
        return [], []
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    output = onet(img_boxes)
    
    faceFeatureModel = OnetFeatures(onet)

    featureOutputs = faceFeatureModel(img_boxes)
    
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    
    featureOutputs = featureOutputs[keep]

    return featureOutputs

#---------------------------------------------------------------------------
# LOAD TEST DATASET
#---------------------------------------------------------------------------

test_data_filelist = sorted(os.listdir('../Dataset/emotiw/test/test_shared/'))

for i in test_data_filelist:
    if i[0] != 't':
        test_data_filelist.remove(i)

print(len(test_data_filelist))

#---------------------------------------------------------------------------
# EXTRACT FACE FEATURES
#---------------------------------------------------------------------------

for i in range(len(test_data_filelist)):
    filename = test_data_filelist[i]
    filename = filename[:-4]
    image = Image.open('../Dataset/emotiw/test/test_shared/'+filename+'.jpg')

    print(filename)
    try:
        
        if os.path.isfile(processed_dataset_path + filename + '.npz'):
            print(filename + ' Already present')
            continue
            
        features = get_face_features(image)
        
        if (type(features)) == tuple:
            with open('hello.text', 'a') as f:
                f.write(filename)
            continue
            
        features = features.data.numpy()

        if features.size == 0:
            print('MTCNN model handling empty face condition at ' + filename)
        np.savez(processed_dataset_path + filename , a=features)   
            
    except ValueError:
        print('No faces detected for ' + filename + ". Also MTCNN failed.")
        np.savez(processed_dataset_path + filename , a=np.zeros(1))
        continue