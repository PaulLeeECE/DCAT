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


#Image size after processing
SIZE_WIDTH = 224
SIZE_HEIGHT = 224
#the directory save you preprocess data
saveDir = '../data/MSDataSet_process/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
#The dataset
#matFile = '/home/share/fating/OriginalDataset/MSDataSet/data/annotations'
#ImagePath = '/home/share/fating/OriginalDataset/MSDataSet/images/'

matFile = "/mnt/data0/aimm_lynn/group_affect/Most-important-person/MS DataSet/data/annotations.mat"
ImagePath = "/mnt/data0/aimm_lynn/group_affect/Most-important-person/MS DataSet/images/"

data = scio.loadmat(matFile)
train = data['train']
print("train: {}".format(train.shape))
val = data['val']
print("val: {}".format(val.shape))
test = data['test']
print("test: {}".format(test.shape))

num_train = train.shape[1]
num_val = val.shape[1]
num_test = test.shape[1]
trainSet=[]
testSet=[]
valSet=[]

#process train set
for i in range(num_train):
    name = train[0,i]['name'][0]
    print('Converting the %dth Image %s' % (i,name))
    width = train[0,i]['width'][0][0]
    height = train[0,i]['height'][0][0]
    foldName = name.split('.')[0]   
    trainSet.append(foldName)
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
    NumFace = train[0,i]['Face'].shape[1]
    img = Image.open(ImagePath+'train/'+name).convert('RGB')
    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = train[0,i]['Face'][0,j]['rect']
        Rect.resize(4,)
        train[0,i]['Face'][0,j]['label'].resize(1,)
        label = int(train[0,i]['Face'][0,j]['label'][0])
        label = 1 if label>1 else label
        x, y, w, h = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3])
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
#process val set
for i in range(num_val):
    print('Converting the %dth Image' % (i))
    name = val[0,i]['name'][0]
    width = val[0,i]['width'][0][0]
    height = val[0,i]['height'][0][0]
    foldName = name.split('.')[0]   
    valSet.append(foldName)
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
    NumFace = val[0,i]['Face'].shape[1]
    img = Image.open(ImagePath+'val/'+name).convert('RGB')
    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = val[0,i]['Face'][0,j]['rect']
        Rect.resize(4,)
        val[0,i]['Face'][0,j]['label'].resize(1,)
        label = int(val[0,i]['Face'][0,j]['label'][0])
        label = 1 if label>1 else label
        x, y, w, h = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3])
        xMin = max(1,int(x-w/2))
        xMax = min(width,int(x+w/2))
        yMin = max(1, int(y-h/2))
        yMax = min(height, int(y+h/2))

        c_xMin = int(max(1,int(x-3*w)))
        c_xMax = int(min(width, int(x+3*w)))
        c_yMin = max(1,y-2*h)
        c_yMax = min(height, y+6*h)
        TempFace = img.crop([xMin,yMin,xMax,yMax]).resize([224,224])
        No_Face = '0' + str(j)
        FaceName = FaceFolderName+'/' + 'Image_' + foldName + \
                   '_Face_' + No_Face[len(No_Face)-2:] + '_Label_' + str(int(label)) + '.jpg'
        TempFace.save(FaceName)
        TempFaceCont = img.crop([c_xMin,c_yMin,c_xMax,c_yMax]).resize([224,224])
        # TempFaceCont = img.crop([c_yMin,c_xMin,c_yMax,c_xMax]).resize([224,224])
        FaceContName = FaceContFolderName+'/' + 'Image_' \
                       + foldName + '_Face_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempFaceCont.save(FaceContName)
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[yMin:yMax, xMin:xMax] = 255
        TempCoor = Image.fromarray(np.uint8(canvas))
        TempCoor = TempCoor.resize([224,224])
        CoorName = CoorFolderName+'/' + 'Image_' \
                       + foldName + '_Coor_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempCoor.save(CoorName)
        ImgName = FullImgFolderName+'/' + 'Image_' \
                       + foldName + '_Img_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        imgCopy.save(ImgName)
#process test set
for i in range(num_test):
    print('Converting the %dth Image' % (i))
    name = test[0,i]['name'][0]
    width = test[0,i]['width'][0][0]
    height = test[0,i]['height'][0][0]
    foldName = name.split('.')[0]   
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
    NumFace = test[0,i]['Face'].shape[1]
    img = Image.open(ImagePath+'test/'+name).convert('RGB')
    img = img.resize([width,height])
    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = test[0,i]['Face'][0,j]['rect']
        Rect.resize(4,)
        test[0,i]['Face'][0,j]['label'].resize(1,)
        label = int(test[0,i]['Face'][0,j]['label'][0])
        label = 1 if label>1 else label
        x, y, w, h = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3])
        xMin = max(1,int(x-w/2))
        xMax = min(width,int(x+w/2))
        yMin = max(1, int(y-h/2))
        yMax = min(height, int(y+h/2))

        c_xMin = int(max(1,int(x-3*w)))
        c_xMax = int(min(width, int(x+3*w)))
        c_yMin = max(1,y-2*h)
        c_yMax = min(height, y+6*h)
        TempFace = img.crop([xMin,yMin,xMax,yMax]).resize([224,224])
        No_Face = '0' + str(j)
        FaceName = FaceFolderName+'/' + 'Image_' + foldName + \
                   '_Face_' + No_Face[len(No_Face)-2:] + '_Label_' + str(int(label)) + '.jpg'
        TempFace.save(FaceName)
        TempFaceCont = img.crop([c_xMin,c_yMin,c_xMax,c_yMax]).resize([224,224])
        # TempFaceCont = img.crop([c_yMin,c_xMin,c_yMax,c_xMax]).resize([224,224])
        FaceContName = FaceContFolderName+'/' + 'Image_' \
                       + foldName + '_Face_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempFaceCont.save(FaceContName)
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[yMin:yMax, xMin:xMax] = 255
        TempCoor = Image.fromarray(np.uint8(canvas))
        TempCoor = TempCoor.resize([224,224])
        CoorName = CoorFolderName+'/' + 'Image_' \
                       + foldName + '_Coor_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempCoor.save(CoorName)
        ImgName = FullImgFolderName+'/' + 'Image_' \
                       + foldName + '_Img_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        imgCopy.save(ImgName)

#print("trainSet: ")
#print(trainSet)

np.save('../data/MSindex.npy', {'train': trainSet, 'val': valSet, 'test': testSet})
