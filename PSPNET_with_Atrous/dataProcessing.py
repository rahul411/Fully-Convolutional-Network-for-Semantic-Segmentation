import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import skimage
import cv2 
import os
import scipy.io as sio
from scipy.ndimage import imread
# from vgg_preprocessing import (_mean_image_subtraction,
#                                              _R_MEAN, _G_MEAN, _B_MEAN)
# import vgg_preprocessing


network_name = 'resnet_v1_101'
# batch_size = 10
checkpoint = '../../resnet_v1_101/resnet_v1_101.ckpt'
layer_names = ['resnet_v1_101/block1']
NUM_CLASSES = 21
cityscape_directory = '../../../cityscape_dataset/'
imgs_directory = '../../../VOCdevkit/VOC2012/JPEGImages/'
imgs_aug_directory = '../../../benchmark_RELEASE/dataset/img/'
labels_directory = '../../../benchmark_RELEASE/dataset/cls/'
segmentation_directory = '../../../VOCdevkit/VOC2012/SegmentationClass/'
VGG_MEAN = [103.939, 116.779, 123.68]
slim = tf.contrib.slim

label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32)]
                # 18 = bicycle

def decode_labels(mask, img_shape, num_classes):
    color_table = label_colours

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (img_shape[0], img_shape[1], img_shape[2], 3))
    
    return pred

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch

def load_labels(path):
    label = sio.loadmat(path)
    return np.expand_dims(label['GTcls']['Segmentation'][0][0].astype(np.float32),axis=2)

def random_crop(image, label):
    random_crop_with_mask = lambda image, label: tf.unstack(
        tf.random_crop(tf.concat((image, label), axis=-1), _crop_shape), axis=-1)
    channel_list = random_crop_with_mask(image, label)
    image = tf.transpose(channel_list[:-1], [1,2,0])
    label = tf.expand_dims(channel_list[-1], axis=-1)
    return image, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    label = tf.cast(label, dtype=tf.float32)
    image = tf.cast(image, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop

def image_mirroring(img, label):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    
    return img, label

def image_scaling(img, label):
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, label

def random_flip(image, label):
    im_la = tf.unstack(tf.image.random_flip_left_right(tf.concat((image, label), axis=-1)),num=24, axis=-1)
    image = tf.transpose(im_la[:3], [1,2,0])
    label = tf.transpose(im_la[3:], [1,2,0])
    return (image, label)

#This function crops the image into 224x224 size. This is done because the VGG model requires the image to be resized.
def crop_image(x, target_height=224, target_width=224):
    # x = x.decode('utf8')
    # p = Path(x)
    # image = cv2.imread(str(p.resolve(x)))
    image = imread(x)
    return cv2.resize(image,(target_width,target_height))
   
def read_and_crop_images(filename, filetype, target_width=224,target_height=224):
    image_string = tf.read_file(filename)
    if filetype == 'jpeg':
      image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    else:
      image_decoded = tf.image.decode_image(image_string, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image_decoded,target_width,target_height)

    return image

def read_images(filename, filetype, channels=1):
    image_string = tf.read_file(filename)
    if filetype == 'jpeg':
      image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
    else:
      image_decoded = tf.image.decode_image(image_string, channels=channels)

    return image_decoded

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

def pascal_palette_for_cityscape():
    ignore_label = 255
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    return id_to_trainid

def convert_to_train_ids(arr_3d):
    arr_3d = np.squeeze(arr_3d)
    #print(arr_3d.shape)
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = pascal_palette_for_cityscape()
    for c, i in palette.items():
        m = np.where(arr_3d == c)
        arr_2d[m] = i
    return np.expand_dims(arr_2d,axis=2)

#For VOC
def createDataset_for_augmentedData(data_type='train'):
    
    train_imgs = []
    val_imgs = []
    with open('../../../benchmark_RELEASE/dataset/train.txt','r') as file:
        
        for line in file:
            train_imgs.append(line.replace('\n',''))
        print(len(train_imgs))

    with open('../../../benchmark_RELEASE/dataset/val.txt','r') as file:
        
        for line in file:
            val_imgs.append(line.replace('\n',''))
        print(len(val_imgs))
    if data_type == 'train':
        train_images = list(map(lambda img_path: imgs_aug_directory + img_path + '.jpg', train_imgs))
        train_labels = list(map(lambda img_path: labels_directory + img_path + '.mat', train_imgs))
        train_data = tf.data.Dataset.from_tensor_slices((train_images,train_labels))

        print('done preparing training data')
        return train_data
    else:
        val_images = list(map(lambda img_path: imgs_aug_directory + img_path + '.jpg', val_imgs))
        val_labels = list(map(lambda img_path: labels_directory + img_path + '.mat', val_imgs))
        val_data = tf.data.Dataset.from_tensor_slices((val_images,val_labels))

        print('done preparing validation data')
        return val_data

#For cityscape
def createDataset_for_cityscape(data_type='train'):
    
    train_imgs = []
    val_imgs = []
    train_labels = []
    val_labels = []
    with open('cityscapes_train_list.txt','r') as file:
        
        for line in file:
            img, label = line.replace('\n','').split(' ')
            train_imgs.append(img)
            train_labels.append(label)
        print(len(train_imgs))

    with open('cityscapes_val_list.txt','r') as file:
        
        for line in file:
            img, label =  line.replace('\n','').split(' ')
            val_imgs.append(img)
            val_labels.append(label)
        print(len(val_imgs))

    if data_type == 'train':
        train_images = list(map(lambda img_path: cityscape_directory + img_path , train_imgs))
        train_labels = list(map(lambda img_path: cityscape_directory + img_path , train_labels))
        train_data = tf.data.Dataset.from_tensor_slices((train_images,train_labels))

        print('done preparing training data')
        return train_data
    else:
        val_images = list(map(lambda img_path: cityscape_directory + img_path , val_imgs))
        val_labels = list(map(lambda img_path: cityscape_directory + img_path , val_labels))
        val_data = tf.data.Dataset.from_tensor_slices((val_images,val_labels))

        print('done preparing validation data')
        return val_data

def createDataset(data_type='train'):
    
    train_imgs = []
    val_imgs = []
    with open('../../../VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt','r') as file:
        
        for line in file:
            train_imgs.append(line.replace('\n',''))
        print(len(train_imgs))

    with open('../../../VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt','r') as file:
        
        for line in file:
            val_imgs.append(line.replace('\n',''))
        print(len(val_imgs))
    if data_type == 'train':
        train_images = list(map(lambda img_path: imgs_directory + img_path + '.jpg', train_imgs[:10]))
        train_labels = list(map(lambda img_path: segmentation_directory + img_path + '.png', train_imgs[:10]))
        train_data = tf.data.Dataset.from_tensor_slices((train_images,train_labels))

        print('done preparing training data')
        return train_data
    else:
        val_images = list(map(lambda img_path: imgs_directory + img_path + '.jpg', val_imgs))
        val_labels = list(map(lambda img_path: segmentation_directory + img_path + '.png', val_imgs))
        val_data = tf.data.Dataset.from_tensor_slices((val_images,val_labels))

        print('done preparing validation data')
        return val_data
        
def resnet_features(image,layer_name):
    image_float = tf.to_float(image, name='ToFloat')
    #processed_image = _mean_image_subtraction(image_float,
     #                                          [_R_MEAN, _G_MEAN, _B_MEAN])
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image_float)
        
    processed_image = tf.concat(axis=3, values=[
        red - VGG_MEAN[2],
        green - VGG_MEAN[1],
        blue - VGG_MEAN[0],
        ])
   # processed_image = vgg_preprocessing.preprocess_image(image_float,224,224,is_training=False)
    #processed_image = tf.expand_dims(processed_image,axis=0)
    arg_scope = nets.resnet_v1.resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        logits,ends = nets.resnet_v1.resnet_v1_101(processed_image,num_classes=1000,is_training=False,output_stride=8, global_pool=False)
        return ends[layer_name]
     #   return ends[layer_name] 


