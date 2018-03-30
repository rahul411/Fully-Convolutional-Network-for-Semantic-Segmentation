import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
from dataProcessing import *
import math
import GPUtil
from utils import get_available_gpus, computeIoU

batchSize = 1
img_height = 224
img_width = 224
num_channels = 3
patchSize = 5
learningRate = 0.0001
weight_decay = 1e-4
num_classes = 21
n_epochs = 175
model_path = 'model'
TRAIN_FLAG = True
delta = 63.0/255.0


VGG_MEAN = [103.939, 116.779, 123.68]
data_dict = np.load('../../vgg19.npy', encoding='latin1').item()

print(len(data_dict['fc8']))


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


def conv_layer(bottom, inchannels, outchannels, name):
    # print(name)
    with tf.variable_scope(name) as scope:
        filt, conv_biases = get_var(name)
        
        conv = tf.nn.conv2d(bottom,filt,[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)

        return relu

def fc_layer(bottom, inchannels, outchannels, name):
    with tf.variable_scope(name) as scope:
        weight, conv_biases = get_var(name)

        if name == 'fc6':
        	weight = tf.reshape(weight,[7,7,512,4096])
        else:
        	weight = tf.reshape(weight,[1,1,4096,4096])

        conv = tf.nn.conv2d(bottom,weight,[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)

        return bias

def get_var(name):
    # print(name)
    # print(data_dict[name]['weights'])
    # if name in data_dict:
    init = tf.constant_initializer(value=data_dict[name][0],dtype=tf.float32)
    # else:
    #     init = tf.constant_initializer(value=tf.truncated_normal)
    shape = data_dict[name][0].shape
    filt = tf.get_variable(name="filter", initializer=init, shape=shape)
    # filt = tf.Variable(data_dict[name][0],name=name+'filt')

    init = tf.constant_initializer(data_dict[name][1],
                                       dtype=tf.float32)
    shape = data_dict[name][1].shape
    bias = tf.get_variable(name="biases", initializer=init, shape=shape)
    # bias = tf.Variable(data_dict[name][1],name=name+'bias')

    return filt, bias

def get_variable_with_decay(shape,stddev, name):
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev = stddev), name=name)
    weights_with_decay = tf.multiply(tf.nn.l2_loss(weights), weight_decay)
    # print(weights)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights_with_decay)

    return weights

def get_bias_variable(shape,name, constant=0.0):
    return tf.Variable(tf.constant(constant,shape=shape),name=name)

def score_layer(bottom, num_classes, name):
    with tf.variable_scope(name) as scope:
    	# print(bottom)
        infeatures = bottom.get_shape()[3].value
        shape = [1,1,infeatures,num_classes]

        stddev = (2/infeatures)**0.5
        weights = get_variable_with_decay(shape,stddev, name)
        # print(weights)

        biases = get_bias_variable([num_classes],name, 0.0)

        conv = tf.nn.conv2d(bottom,weights,[1,1,1,1], padding='SAME')
        bias_add = tf.nn.bias_add(conv,biases)

        return bias_add 

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
    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,shape=weights.shape)


def upscore_layer(bottom, name, shape, num_classes,ksize, stride):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name) as scope:
        infeatures = bottom.get_shape()[3].value
        weight_shape = [ksize,ksize,num_classes,infeatures]

        output_shape = tf.stack([shape[0], shape[1], shape[2], num_classes])

        weights = get_deconv_filter(weight_shape)
        deconv = tf.nn.conv2d_transpose(bottom,weights,output_shape,strides=strides,padding='SAME')

        return deconv

def model(img):

    # tf.summary.image('train_images',img)
    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=img)
        
    bgr = tf.concat(axis=3, values=[
        blue,
        green,
        red,
        ])

    conv1_1 = conv_layer(bgr,3,64,'conv1_1')
    conv1_2 = conv_layer(conv1_1,64,64,'conv1_2')
    pool1 = tf.nn.max_pool(conv1_2,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool1')

    conv2_1 = conv_layer(pool1,64,128,'conv2_1')
    conv2_2 = conv_layer(conv2_1,128,128,'conv2_2')
    pool2 = tf.nn.max_pool(conv2_2,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool2') 

    conv3_1 = conv_layer(pool2,128,256,'conv3_1')
    conv3_2 = conv_layer(conv3_1,256,256,'conv3_2')
    conv3_3 = conv_layer(conv3_2,256,256,'conv3_3')
    pool3 = tf.nn.max_pool(conv3_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool3')

    conv4_1 = conv_layer(pool3,256,512,'conv4_1')
    conv4_2 = conv_layer(conv4_1,512,512,'conv4_2')
    conv4_3 = conv_layer(conv4_2,512,512,'conv4_3')
    pool4 = tf.nn.max_pool(conv4_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool4')

    conv5_1 = conv_layer(pool4,512,512,'conv5_1')
    conv5_2 = conv_layer(conv5_1,512,512,'conv5_2')
    conv5_3 = conv_layer(conv5_2,512,512,'conv5_3')
    pool5 = tf.nn.max_pool(conv5_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool5')

    fc_6 = fc_layer(pool5, 25088, 4096, 'fc6')
    fc_6 = tf.nn.relu(fc_6)
    fc_6 = tf.nn.dropout(fc_6,0.5)

    fc_7 = fc_layer(fc_6, 4096, 4096, 'fc7') 
    fc_7 = tf.nn.relu(fc_7)
    fc_7 = tf.nn.dropout(fc_7,0.5)

    s_layer_8 = score_layer(pool3, num_classes, 'score_layer_pool3')

    s_layer_16 = score_layer(pool4, num_classes, 'score_layer_pool4')

    s_layer_32 = score_layer(fc_7, num_classes, 'score_layer_fc7')

    upscore_32 = upscore_layer(s_layer_32, 'upscore_s_layer2', tf.shape(pool4), num_classes, ksize=4, stride=2)

    fuse_layer_16 = tf.add(s_layer_16, upscore_32)

    upscore_16 = upscore_layer(fuse_layer_16,'upscore_fuse_layer16', tf.shape(pool3), num_classes, ksize=4, stride=2)

    fuse_layer_8 = tf.add(upscore_16, s_layer_8)

    upscore_8 = upscore_layer(fuse_layer_8,'upscore_fuse_layer8', tf.shape(bgr), num_classes, ksize=16, stride=8)

    pred = tf.argmax(upscore_8,axis=3)
    # tf.summary.image('grayScale_prediction',tf.cast(tf.expand_dims(tf.multiply(pred,tf.constant(10, dtype=tf.int64)),3),tf.float16))

    return pred, upscore_8


graph = tf.Graph()
with graph.as_default():
    gpus = get_available_gpus()
    m = lambda x: tf.py_func(convert_from_color_segmentation, [x], tf.float32)
    readImages = lambda x: tf.py_func(crop_image, [x], tf.float32)
    mIoU = lambda x,y : tf.py_func(computeIoU,[x,y], tf.float32)

    train_dataset = createDataset()
    train_dataset = train_dataset.map(lambda x, y: (tf.image.per_image_standardization(tf.image.random_brightness(readImages(x), delta)), m(readImages(y))))
    train_dataset = train_dataset.shuffle(buffer_size= 10)
    train_dataset = train_dataset.batch(batchSize)

    val_dataset = createDataset()
    val_dataset = val_dataset.map(lambda x, y: (tf.image.per_image_standardization(tf.image.random_brightness(readImages(x), delta)), m(readImages(y))))
    val_dataset = val_dataset.shuffle(buffer_size= 10)
    val_dataset = val_dataset.batch(batchSize)

    # iterator = dataset.make_initializable_iterator()
    # create TensorFlow Iterator object
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                   train_dataset.output_shapes)

    # create two initialization ops to switch between the datasets
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(val_dataset)

    images, labels = iterator.get_next()
    with tf.variable_scope():
        opt = tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=0.9)
    
    tower_grads = []
    tower_loss = []
    iou = []
    for i in range(len(gpus)):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('Tower_%d' % (i)) as scope:

                images_slice = get_slice(images,i,len(gpus))
                labels_slice = get_slice(labels,i,len(gpus))
                pred, logits = model(images_slice)
                with tf.device('/cpu:0'):
                    iou.append(mIoU(pred, labels_slice))
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels_slice))
                if TRAIN_FLAG:
                    tower_grads.append(opt.compute_gradients(loss))
                tower_loss.append(loss)
                tf.get_variable_scope().reuse_variables()

    total_loss = tf.add_n(tower_loss)
    if TRAIN_FLAG:
        tf.summary.image('train_images',images_slice)
        tf.summary.image('grayScale_prediction',tf.cast(tf.expand_dims(tf.multiply(pred,tf.constant(10, dtype=tf.int64)),3),tf.float16))
        tf.summary.image('grayscale_groundtruth',tf.cast(tf.expand_dims(tf.multiply(tf.argmax(labels_slice,axis=3),tf.constant(10,dtype=tf.int64)),3),tf.float16))
        tf.summary.scalar('Train_step_loss',tf.add_n(tower_loss))
        tf.summary.scalar('Train_mIoU',tf.add_n(iou))
    else:
        tf.summary.scalar('val_mIoU',tf.add_n(iou))
    grads = average_gradients(tower_grads)
    optimizer = opt.apply_gradients(grads)
    summary = tf.summary.merge_all()

with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver(max_to_keep=50)

    
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter("output", sess.graph)
    
    for epoch in range(n_epochs):
        
        sess.run(training_init_op)

        print(GPUtil.showUtilization())
        TRAIN_FLAG = True
        while True:
            try:

                step_loss, _, summary_ = sess.run([total_loss, optimizer, summary])
                print(step_loss)
                writer.add_summary(summary_)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
        if epoch%10 == 0:
            saver.save(sess, os.path.join(model_path,'model'),global_step=epoch)

        sess.run(validation_init_op)
        TRAIN_FLAG = False
        while True:
            try:

                val_step_loss,_summary = sess.run([total_loss, summary])
                writer.add_summary(_summary)
            except tf.errors.OutOfRangeError:
                print("End of validation dataset.")
                break
        writer.flush()
    writer.close()
        