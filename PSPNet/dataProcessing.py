import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import skimage
import cv2 
import os
import GPUtil
from scipy.ndimage import imread
from preprocessing.vgg_preprocessing import (_mean_image_subtraction,
                                            _R_MEAN, _G_MEAN, _B_MEAN)
from preprocessing import vgg_preprocessing


network_name = 'resnet_v1_101'
# batch_size = 10
checkpoint = '../../resnet_v1_101/resnet_v1_101.ckpt'
layer_names = ['resnet_v1_101/block1']
NUM_CLASSES = 21
imgs_directory = '../../VOCdevkit/VOC2012/JPEGImages/'
segmentation_directory = '../../VOCdevkit/VOC2012/SegmentationClass/'
VGG_MEAN = [103.939, 116.779, 123.68]
slim = tf.contrib.slim

def random_crop(self, image, label):
    random_crop_with_mask = lambda image, label: tf.unstack(
        tf.random_crop(tf.concat((image, label), axis=-1), self._crop_shape), axis=-1)
    channel_list = random_crop_with_mask(image, label)
    image = tf.transpose(channel_list[:-1], [1,2,0])
    label = tf.expand_dims(channel_list[-1], axis=-1)
    return image, label

def random_flip(image, label):
    im_la = tf.unstack(tf.image.random_flip_left_right(tf.concat((image, label), axis=-1)),num=24, axis=-1)
    image = tf.transpose(im_la[:3], [1,2,0])
    label = tf.transpose(im_la[3:], [1,2,0])
    return (image, label)

#This function crops the image into 224x224 size. This is done because the VGG model requires the image to be resized.
def crop_image(x, target_height=512, target_width=512):
    # x = x.decode('utf8')
    # p = Path(x)
    # image = cv2.imread(str(p.resolve(x)))
    image = imread(x)
    return np.array(cv2.resize(image,(target_width,target_height)), dtype=np.float32)
   
def read_and_crop_images(filename, filetype, target_width=512,target_height=512):
    image_string = tf.read_file(filename)
    if filetype == 'jpeg':
      image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    else:
      image_decoded = tf.image.decode_image(image_string, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image_decoded,target_width,target_height)

    return image

def input_parser(label_path):
    
    label_img = imread(label_path)
    label_img = convert_from_color_segmentation(label_img)
    
    return label_img

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
        train_images = list(map(lambda img_path: imgs_directory + img_path + '.jpg', train_imgs[:10]))
        train_labels = list(map(lambda img_path: segmentation_directory + img_path + '.png', train_imgs[:10]))
        train_data = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
        print(GPUtil.showUtilization())

        print('done preparing training data')
        return train_data
    else:
        val_images = list(map(lambda img_path: imgs_directory + img_path + '.jpg', val_imgs[:10]))
        val_labels = list(map(lambda img_path: segmentation_directory + img_path + '.png', val_imgs[:10]))
        val_data = tf.data.Dataset.from_tensor_slices((val_images,val_labels))
        print(GPUtil.showUtilization())

        print('done preparing validation data')
        return val_data
        
def resnet_features(image,layer_name):
    image_float = tf.to_float(image, name='ToFloat')
    # processed_image = _mean_image_subtraction(image_float,
    #                                           [_R_MEAN, _G_MEAN, _B_MEAN])
    red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=image_float)
        
    processed_image = tf.concat(axis=2, values=[
        red - _R_MEAN,
        green - _G_MEAN,
        blue - _B_MEAN,
        ])
    # processed_image = vgg_preprocessing.preprocess_image(image_float,224,224,is_training=False)
    processed_image = tf.expand_dims(processed_image,axis=0)
    arg_scope = nets.resnet_v1.resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        logits,ends = nets.resnet_v1.resnet_v1_101(processed_image,num_classes=1000,is_training=False,output_stride=8, global_pool=False)
        return tf.squeeze(ends[layer_name])


