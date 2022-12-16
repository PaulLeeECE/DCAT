import os
import numpy as np
import random

import skimage
from skimage import data, io
# import skimage.io
import skimage.transform
import skimage.color
import pdb
import pickle
import fnmatch
from PIL import Image
import scipy.io as scio

import cv2
from mtcnn import MTCNN
import glob

detector = MTCNN()

#Image size after processing
SIZE_WIDTH = 224
SIZE_HEIGHT = 224
#the directory save you preprocess data
saveDir = '../new_data/GroupEmoW_process/GroupEmoW_Val/Negative/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
#The dataset
#matFile = "/mnt/data0/aimm_lynn/group_affect/Most-important-person/MS DataSet/data/annotations.mat"
#ImagePath = "/mnt/data0/aimm_lynn/group_affect/Most-important-person/MS DataSet/images/"

img_path = "/mnt/data0/aimm_lynn/group_affect/datasets/GroupEmoW/GroupEmoW_Val/Negative/"
imgDir = glob.glob(img_path+"*")

print(len(imgDir))

#imgDir = []
#imgDir.append("/mnt/data0/aimm_lynn/group_affect/Most-important-person/MS DataSet/images/test/0001.jpg")

'''data = scio.loadmat(matFile)
train = data['train']
print("train: {}".format(train.shape))
val = data['val']
print("val: {}".format(val.shape))
test = data['test']
print("test: {}".format(test.shape))'''

'''num_train = train.shape[1]
num_val = val.shape[1]
num_test = test.shape[1]'''


trainSet=[]
testSet=[]
valSet=[]

#process train set
count = 0
for imgd in imgDir:
    count = count + 1;
    name = imgd.split('/')[-1]
    
    print('Converting the %dth Image %s' % (count, name))
    
    imr = cv2.imread(imgd)
    image = cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    
    width = imr.shape[1]
    #print("width: {}".format(width))
    height = imr.shape[0]
    #print("height: {}".format(height))
    foldName = name.split('.')[0].split('_')[-1]
    #print("foldName: {}".format(foldName))
    testSet.append(foldName)
    FaceFolderName = saveDir + 'Image_'+foldName+ '/Face'
    
    if not os.path.exists(FaceFolderName):
        os.makedirs(FaceFolderName)
    FaceContFolderName = saveDir + 'Image_'+foldName + '/FaceCont'
    if not os.path.exists(FaceContFolderName):
        os.makedirs(FaceContFolderName)
    CoorFolderName = saveDir + 'Image_'+foldName + '/Coordinate'
    if not os.path.exists(CoorFolderName):
        os.makedirs(CoorFolderName)
    FullImgFolderName = saveDir + 'Image_'+foldName + '/Image'
    if not os.path.exists(FullImgFolderName):
        os.makedirs(FullImgFolderName)
    
    NumFace = len(result)
    img = Image.open(imgd).convert('RGB')
    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    
    for j in range(NumFace):
        Rect = result[j]['box']
        label = 0
        x, y, w, h = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3])
        x = x+w/2
        y = y+h/2
        xMin = max(1,int(x-w/2))
        xMax = min(width,int(x+w/2))
        yMin = max(1, int(y-h/2))
        yMax = min(height, int(y+h/2))

        c_xMin = int(max(1,int(x-3*w)))
        c_xMax = int(min(width, int(x+3*w)))
        c_yMin = max(1,y-2*h)
        c_yMax = min(height, y+6*h)
        TempFace = img.crop([xMin,yMin,xMax,yMax]).resize([SIZE_WIDTH,SIZE_HEIGHT])
        No_Face = '0' + str(j)
        FaceName = FaceFolderName+'/' + 'Image_' + foldName + \
                   '_Face_' + No_Face[len(No_Face)-2:] + '_Label_' + str(int(label)) + '.jpg'
        TempFace.save(FaceName)
        
        TempFaceCont = img.crop([c_xMin,c_yMin,c_xMax,c_yMax]).resize([SIZE_WIDTH,SIZE_HEIGHT])
        # TempFaceCont = img.crop([c_yMin,c_xMin,c_yMax,c_xMax]).resize([224,224])
        FaceContName = FaceContFolderName+'/' + 'Image_' \
                       + foldName + '_Face_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempFaceCont.save(FaceContName)
        
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[yMin:yMax, xMin:xMax] = 255
        TempCoor = Image.fromarray(np.uint8(canvas))
        TempCoor = TempCoor.resize([SIZE_WIDTH,SIZE_HEIGHT])
        CoorName = CoorFolderName+'/' + 'Image_' \
                       + foldName + '_Coor_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempCoor.save(CoorName)
        ImgName = FullImgFolderName+'/' + 'Image_' \
                       + foldName + '_Img_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        imgCopy.save(ImgName)
        
testSet.sort()

np.save('../new_data/GroupEmoW_process/GroupEmoW_Val/Neg_index.npy', {'train': trainSet, 'val': valSet, 'test': testSet})

print("testSet: ")
print(testSet)

