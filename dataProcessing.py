import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
import skimage
import cv2 
import os
import GPUtil

batchSize = 2
NUM_CLASSES = 21
imgs_directory = '../../VOCdevkit/VOC2012/JPEGImages/'
segmentation_directory = '../../VOCdevkit/VOC2012/SegmentationClass/'

# img = cv2.imread(segmentation_directory+'2007_000032.png')
# print(img.shape)


#This function crops the image into 224x224 size. This is done because the VGG model requires the image to be resized.
def crop_image(x, target_height=224, target_width=224):
    # print(x)
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

    return cv2.resize(resized_image, (target_height, target_width))

def input_parser(img_path,label_path):
    # one_hot = tf.one_hot(label,NUM_CLASSES)
    train_img = crop_image(img_path, 224, 224)

    label_img = crop_image(img_path, 224, 224)
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
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

    return palette

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1], NUM_CLASSES), dtype=np.uint8)
    palette = pascal_palette()

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        n = np.zeros(NUM_CLASSES)
        n[i] = 1
        arr_2d[m] = n
    # print(arr_2d.shape)
    return arr_2d
####################################################################################################################

  
def createDataset():
    
    train_imgs = []
    val_imgs = []
    with open('../../VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt','r') as file:
        train_imgs = file.readlines()
        for i in range(len(train_imgs)):
        	train_imgs[i] = train_imgs[i].replace('\n','')
        print(len(train_imgs))

    with open('../../VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt','r') as file:
        val_imgs = file.readlines()
        for i in range(len(val_imgs)):
        	val_imgs[i] = val_imgs[i].replace('\n','')
        print(len(val_imgs))

    train_images = map(lambda img_path: imgs_directory + img_path + '.jpg', train_imgs)
    train_labels = map(lambda img_path: segmentation_directory + img_path + '.png', train_imgs)
    
    print(GPUtil.showUtilization())
    train_data = Dataset.from_tensor_slices((train_images,train_labels))
    print(GPUtil.showUtilization())

    # train_data = train_data.map(
    # lambda filename, label: tuple(tf.py_func(input_parser, [train_images, train_labels], [tf.float32, tf.float32])))


    # train_images = map(crop_image,train_images)
    # train_images = np.array(train_images,dtype=np.float32)
    # train_labels = map(crop_image,train_labels)
    # train_labels = map(convert_from_color_segmentation, train_labels)
    # train_labels = np.array(train_labels,dtype=np.float32)

    # print(train_labels)
    # print(train_images)

    # train_data = Dataset.from_tensor_slices((train_images,train_labels))
    # train_data = train_data.batch(batchSize)
    # train_data = (train_images,train_labels)

    print('done preparing training data')

    # val_images = map(lambda img_path: imgs_directory + img_path + '.jpg', val_imgs)
    # val_labels = map(lambda img_path: segmentation_directory + img_path + '.png', val_imgs)

    # val_images = map(crop_image,val_images)
    # val_images = np.array(val_images,dtype=np.float32)
    # val_labels = map(crop_image,val_labels)
    # val_labels = map(convert_from_color_segmentation, val_labels)
    # val_labels = np.array(val_labels,dtype=np.float32)
    
    # val_data = Dataset.from_tensor_slices((val_images,val_labels))

    # val_data = val_data.batch(batchSize)
    # val_data = (val_images,val_labels)

    print('Done loading data')
    
    return train_data    

# t,v = createDataset()

# data = tf.ones([1,3,3,1])

# pool, poolingIndices = tf.nn.max_pool_with_argmax(data,[1,2,2,1],[1,2,2,1],padding='SAME')

# # unravelIndices = np.unravel_index(poolingIndices,[1,3,3,1])

# sess = tf.Session()

# pool, pindices, unravelIndices = sess.run([pool,poolingIndices, unravelIndices])

# print(pindices)
# print(pool)
# print(unravelIndices)