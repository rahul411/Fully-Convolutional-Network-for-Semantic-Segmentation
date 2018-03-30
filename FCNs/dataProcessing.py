import tensorflow as tf
# from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
import skimage
import cv2 
import os
import GPUtil
import matplotlib.pyplot as plt
# from pathutil import Path

batchSize = 2
NUM_CLASSES = 21
imgs_directory = '../../VOCdevkit/VOC2012/JPEGImages/'
segmentation_directory = '../../VOCdevkit/VOC2012/SegmentationClass/'

# img = cv2.imread(segmentation_directory+'2007_000032.png')
# print(img.shape)


#This function crops the image into 224x224 size. This is done because the VGG model requires the image to be resized.
def crop_image(x, target_height=224, target_width=224):
    # x = x.decode('utf8')
    # p = Path(x)
    # image = cv2.imread(srt(p.resolve(x)))
    image = cv2.imread(x)
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return np.array(cv2.resize(resized_image, (target_height, target_width)), dtype=np.float32)

def read_and_crop_images(x, fileType, target_width=224,target_height=224):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)

    image = tf.image.resize_image_with_crop_or_pad(image_decoded,target_width,target_height)

    return image

def input_parser(img_path,label_path):
    # one_hot = tf.one_hot(label,NUM_CLASSES)
    train_img = crop_image(img_path, 224, 224)

    label_img = crop_image(label_path, 224, 224)
    label_img = convert_from_color_segmentation(label_img)
    # img_file = tf.read_file(img_path)
    # img_decoded = tf.image.decode_image(img_file, channels=3)
    # img_decoded = tf.image.resize_images(img_decoded,[224,224,3])

    # label_file = tf.read_file(label_path)
    # label_decoded = tf.image.decode_image(label_file, channels=3)
    # label_decoded = tf.image.resize_images(label_decoded,[224,224,3])

    return train_img, label_img

#############Taken this code from https://github.com/martinkersner/train-DeepLab/blob/master/utils.py################
def pascal_classes():
      classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4,
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8,
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12,
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted-plant' : 16,
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv/monitor'   : 20}

      return classes

def pascal_palette():
    palette = {(  0,   0,   0) : 0 ,
             (0,   0,   128) : 1 ,
             (  0, 128,   0) : 2 ,
             (0, 128,   128) : 3 ,
             (  128,   0, 0) : 4 ,
             (128,   0, 128) : 5 ,
             (  128, 128, 0) : 6 ,
             (128, 128, 128) : 7 ,
             ( 0,   0,   64) : 8 ,
             (0,   0,   192) : 9 ,
             ( 0, 128,   64) : 10,
             (0, 128,   192) : 11,
             ( 128,   0, 64) : 12,
             (128,   0, 192) : 13,
             ( 128, 128, 64) : 14,
             (128, 128, 192) : 15,
             (  0,  64,   0) : 16,
             (0,  64,   128) : 17,
             (  0, 192,   0) : 18,
             (0, 192,   128) : 19,
             (  128,  64, 0) : 20 }
    return palette

def pascal_palette_invert():
    palette_list = pascal_palette().keys()
    palette = ()
  
    for color in palette_list:
        palette += color

    return palette

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1], NUM_CLASSES), dtype=np.float32)
    palette = pascal_palette()
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        n = np.zeros(NUM_CLASSES)
        n[i] = 1
        arr_2d[m] = n
    # print(arr_2d.shape)
    # print('convert',np.amax(np.argmax(arr_2d,axis=2)))
    # if np.amax(np.argmax(arr_2d,axis=2)) == 0:
    #     plt.imshow(arr_3d)
    #     plt.show()
    return np.array(arr_2d)
####################################################################################################################

  
def createDataset(data_type='train'):
    
    train_imgs = []
    val_imgs = []
    with open('../../VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt','r') as file:
        
        for line in file:
        	train_imgs.append(line.replace('\n',''))
        print(len(train_imgs))

    with open('../../VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt','r') as file:
        
        for line in file:
        	val_imgs.append(line.replace('\n',''))
        print(len(val_imgs))
    if data_type == 'train':
        train_images = list(map(lambda img_path: imgs_directory + img_path + '.jpg', train_imgs))
        train_labels = list(map(lambda img_path: segmentation_directory + img_path + '.png', train_imgs))
        train_data = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
        print(GPUtil.showUtilization())

        print('done preparing training data')
        return train_data
    else:
        val_images = list(map(lambda img_path: imgs_directory + img_path + '.jpg', val_imgs))
        val_labels = list(map(lambda img_path: segmentation_directory + img_path + '.png', val_imgs))
        val_data = tf.data.Dataset.from_tensor_slices((val_images,val_labels))
        print(GPUtil.showUtilization())

        print('done preparing training data')
        return val_data
        
