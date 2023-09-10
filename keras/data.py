import torch
import struct
import numpy as np
import cv2

    
def getTarget_PNG(list: list, i:int):
    targetList = [0, 0, 0, 0, 0, 0, 0]
    target = int(list[i].split(".")[0])
    targetList[target] = 1
    targetTensor = torch.tensor(targetList)
    return targetTensor

def img_into_array(path):
    cv2.imread(path)
    return

def readBinaryData(path: str):
    with open(path ,'rb') as photos:
        magic, size = struct.unpack(">II", photos.read(8)) #first 8 bits are magic number (random number) and the size of the Data
        nrows, ncols = struct.unpack(">II", photos.read(8))
        data = np.fromfile(photos, dtype=np.dtype(np.uint8).newbyteorder('>'))
        return (data.reshape((size, nrows * ncols, 1))) / 255

def readBinaryLabels(path: str):
    with open(path ,'rb') as labels:
        magic, numOfItems = struct.unpack(">II", labels.read(8)) #first 8 bits are magic number (random number) and the size of the labels
        return np.fromfile(labels, dtype=np.dtype(np.uint8).newbyteorder('>'))


def vectorizeResult(index: int):
    list = np.zeros((10, 1))
    list[index] = 1.0
    return list

def getPrepredData(dataPath:str, labelpath:str):
    # prepered data is a list of (img, desired output)
    imgData = readBinaryData(dataPath)
    labelData = readBinaryLabels(labelpath)
    if len(labelData) == len(imgData):
        preperedData = []
        for index in range(len(labelData)):
            preperedData.append((np.array(imgData[index]), vectorizeResult(labelData[index])))

    return preperedData

