import tensorflow as tf
import numpy as np
import GPUtil
import PIL
import os
import math
from utils import *
from dataProcessing import *

layerName = 'resnet_v1_101/block4'
num_classes = 19
batchSize = 5
img_height = 472
img_width = 472
feature_map_height = img_height//8
feature_map_width = img_width//8 
filter_depth = 2048
weight_decay = 0.0001
momentum = 0.9
learningRate = 0.0001
power = 0.9
maxIter = 30000
delta = 63.0/255.0
TRAIN_FLAG = True
model_path = 'modelpspnet_try2'
checkpoint = '../../../resnet_v1_101/resnet_v1_101.ckpt'

def resize_label(label):
    return tf.image.resize_image_with_crop_or_pad(label,img_width,img_height)


def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, v in grad_and_vars:
                if g is not None:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)

            if len(grads)>0:
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)

        return average_grads

def get_slice(data, idx, parts):
    shape = tf.shape(data)
    size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
    stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
    start = stride * idx
    return tf.slice(data, start, size)


def conv_layer(bottom, inchannels, outchannels, name, relu=True, batch_norm=True):
    # print(name)
    with tf.variable_scope(name) as scope:

        filt = get_variable_with_decay([3,3,inchannels, outchannels], name+'/weights')
        conv_biases = get_bias_variable([outchannels], name + '/biases')
        
        conv = tf.nn.atrous_conv2d(bottom,filt,2,padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        if batch_norm:
            layer = tf.layers.batch_normalization(bias)
        if relu:
            relu_layer = tf.nn.relu(layer)
            return relu_layer

        return bias

def get_variable_with_decay(shape, name):
    initializer = tf.contrib.layers.xavier_initializer()
    weights = tf.get_variable(shape=shape,name=name,initializer=initializer)
    weights_with_decay = tf.multiply(tf.nn.l2_loss(weights), weight_decay)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights_with_decay)

    return weights

def get_bias_variable(shape,name, constant=0.0):
    return tf.Variable(tf.constant(constant,shape=shape),name=name)

def get_deconv_filter(shape):
    width = shape[0]
    height = shape[1]
    f = math.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    if shape == None:
        print('None')
    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,shape=weights.shape)


def upscore_layer(bottom, name, shape, num_classes,ksize, stride, inchannels = 256):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name) as scope:
        weight_shape = [ksize,ksize,num_classes,inchannels]

        output_shape = tf.stack([shape[0], shape[1], shape[2], num_classes])

        weights = get_deconv_filter(weight_shape)
        deconv = tf.nn.conv2d_transpose(bottom,weights,output_shape,strides=strides,padding='SAME')

        return deconv

def spatialPooling(bottom, pooling_dims):
	shape = bottom.get_shape()
	pool_layers = {}
	for i in pooling_dims:
		pool = tf.nn.avg_pool(bottom,[1,math.ceil(feature_map_width/i), math.ceil(feature_map_height/i),1],
								[1, math.ceil(feature_map_width/i), math.ceil(feature_map_height/i),1],padding='SAME',name='spatialpool'+ str(i))

		pool_layers[str(i)] = pool

	return pool_layers

def resize_layer(bottom, size):
    return tf.image.resize_bilinear(bottom, size=size)

def model(features):

        
    features_shape = tf.shape(features)
    #pool_layers = spatialPooling(features,[1,2,3,6])
    pool6 = tf.nn.avg_pool(features,[1,10,10,1],[1,10,10,1],padding='VALID',name='spatialpool6')
    pool3 = tf.nn.avg_pool(features,[1,20,20,1],[1,20,20,1],padding='VALID',name='spatialpooling3')
    pool2 = tf.nn.avg_pool(features,[1,30,30,1],[1,30,30,1],padding='VALID',name='spatialpooling2')
    pool1 = tf.nn.avg_pool(features,[1,59,59,1],[1,59,59,1],padding='VALID',name='spatialpooling1')

    sp_conv_6 = conv_layer(pool6, filter_depth, 512, 'sp_conv_6')

    sp_conv_3 = conv_layer(pool3, filter_depth, 512, 'sp_conv_3')

    sp_conv_2 = conv_layer(pool2, filter_depth, 512, 'sp_conv_2')

    sp_conv_1 = conv_layer(pool1, filter_depth, 512, 'sp_conv_1')


    upscore_layer_6 = resize_layer(sp_conv_6, features_shape[1:3])
    upscore_layer_3 = resize_layer(sp_conv_3, features_shape[1:3])
    upscore_layer_2 = resize_layer(sp_conv_2, features_shape[1:3])
    upscore_layer_1 = resize_layer(sp_conv_1, features_shape[1:3])


    concat_layer_1 = tf.concat([upscore_layer_1, upscore_layer_2, upscore_layer_3, upscore_layer_6, features],axis=3)

    concat_layer_2 = conv_layer(concat_layer_1, 4096, 512, 'concat_layer_2')
    final_layer = conv_layer(concat_layer_2, 512, num_classes, 'final_layer', batch_norm = False, relu = False)
    final_layer = resize_layer(final_layer, tf.stack([tf.constant(img_height),tf.constant(img_width)]))
    pred = tf.argmax(final_layer,axis=3)

    return pred, final_layer



graph = tf.Graph()
with graph.as_default():
    
    createTrainLabel = lambda x: tf.py_func(convert_to_train_ids, [x], tf.uint8)

    train_dataset = createDataset_for_cityscape('train')
    train_dataset = train_dataset.map(lambda x, y: (read_images(x,'png',3), createTrainLabel(read_images(y, 'png',1))))
    train_dataset = train_dataset.map(lambda x, y: (random_crop_and_pad_image_and_labels(x,y,img_height,img_width)))
    train_dataset = train_dataset.map(lambda x, y: (image_mirroring(x,y)))
    # train_dataset = train_dataset.shuffle(buffer_size= 2)
    train_dataset = train_dataset.batch(batchSize)

    # val_dataset = createDataset('val')
    # val_dataset = val_dataset.map(lambda x, y: (tf.image.random_brightness(readImages(x), delta), m(y)))
    # # val_dataset = val_dataset.map(lambda x, y: (resnet_features(x, layerName),y))
    # val_dataset = val_dataset.batch(batchSize)

    # create TensorFlow Iterator object
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                   train_dataset.output_shapes)

    # create two initialization ops to switch between the datasets
    training_init_op = iterator.make_initializer(train_dataset)
    # validation_init_op = iterator.make_initializer(val_dataset)

    features, labels = iterator.get_next()
    features = resnet_features(features, layerName)
    #labels = tf.one_hot(tf.cast(labels,tf.int32),num_classes)
    labels = tf.squeeze(labels,axis=3)
    pred, raw_output = model(features)
    pred_flatten = tf.expand_dims(pred,axis=3)
    pred_flatten = tf.reshape(pred_flatten,[-1,])

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, num_classes])
    #label_proc = prepare_label(labels, tf.stack(raw_output.get_shape()[1:3]), num_classes=num_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(labels, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    logits = tf.gather(raw_prediction, indices)
    pred_flatten = tf.gather(pred_flatten, indices)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=gt))
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_flatten, gt, num_classes=num_classes)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdamOptimizer(learningRate)
        grads = optimizer.compute_gradients(loss)
        # grads = list(zip(grads, tf.trainable_variables()))
        # Op to update all variables according to their gradient
        apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
    # Create summaries to visualize weights
    #for var in tf.trainable_variables():
    #    tf.summary.histogram(var.name, var)
    # Summarize all gradients
    #for grad, var in grads:
    #    if grad is not None:
    #         tf.summary.histogram(var.name + '/gradient', grad)
    tf.summary.scalar('features',tf.reduce_mean(features))
    if TRAIN_FLAG: 
        display_label = tf.where(tf.equal(labels,255),tf.zeros(tf.shape(labels),dtype=labels.dtype), labels)
        tf.summary.image('grayScale_prediction',decode_labels(pred,[batchSize, img_height, img_width], num_classes))
        tf.summary.image('grayscale_groundtruth',decode_labels(display_label,[batchSize, img_height, img_width], num_classes))

        tf.summary.scalar('step_loss',loss)
        tf.summary.scalar('train_mIOU',mIoU)
    # else:
    #     tf.summary.scalar('val_mIOU',iou)
    summary = tf.summary.merge_all()


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=50)

    writer = tf.summary.FileWriter("output_try2", sess.graph)
    restore_vars = [
        var for var in tf.global_variables()
        if var.name.startswith('resnet_v1_101/') and 'Adam' not in var.name
    ]
    saver_resnet = tf.train.Saver(restore_vars)
    saver_resnet.restore(sess, os.path.join(checkpoint))

   # tf.global_variables_initializer().run()
    print(GPUtil.showUtilization())
    sess.run(training_init_op)
    for iterNo in range(maxIter):
        TRAIN_FLAG = True
        try :
            iou_update, steploss, _, summary_ = sess.run([update_op, loss,apply_grads, summary])
            print('interationNo:',iterNo)
            print(steploss)
            writer.add_summary(summary_)
            #features_,labels_, summary_ = sess.run([features,labels,summary])
            #print(features_.shape, labels_.shape)
            #print(np.mean(features_), np.max(labels_),np.min(labels_))
            #writer.add_summary(summary_)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            sess.run(training_init_op)

        learningRate*= math.pow(1-iterNo/maxIter, power)
        if iterNo%10 == 0:
            writer.flush()
        if iterNo%200 == 0:
            saver.save(sess, os.path.join(model_path, 'model'), global_step=iterNo)
            # sess.run(validation_init_op)
            # TRAIN_FLAG = False
            # while True:
            #     try:
            #         val_step_loss,_summary = sess.run([loss, summary])
            #         writer.add_summary(_summary)
            #     except tf.errors.OutOfRangeError:
            #         print("End of validation dataset.")
            #         sess.run(training_init_op)
            #         break
    writer.close()
