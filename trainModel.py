import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
from dataProcessing import *
import math
import GPUtil

batchSize = 1
img_height = 224
img_width = 224
num_channels = 3
patchSize = 5
learningRate = 0.001
weight_decay = 1e-4
num_classes = 21
n_epochs = 100
model_path = 'model'


VGG_MEAN = [103.939, 116.779, 123.68]
data_dict = np.load('../../vgg19.npy', encoding='latin1').item()

print(len(data_dict['fc8']))

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

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=img)
        
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
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

    s_layer = score_layer(fc_7, num_classes, 'score_layer_fc7')

    upscore = upscore_layer(s_layer, 'upscore', tf.shape(bgr), num_classes, ksize=64, stride=32)

    pred = tf.argmax(upscore,axis=3)

    return pred, tf.nn.softmax(upscore)


# train_data = tf.py_func(createDataset,tf.float32)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32,shape=[None,img_height,img_width,num_channels])
    y = tf.placeholder(tf.float32,shape=[None,img_height,img_width, None])

    pred, logits = model(x)
    # print(logits.dtype)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=y))

    optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)

with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver(max_to_keep=50)

    
    tf.global_variables_initializer().run()

    # features_placeholder = tf.placeholder(train_data[0].dtype, train_data[0].shape)
    # labels_placeholder = tf.placeholder(train_data[1].dtype, train_data[1].shape)
    # dataset = Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    
    dataset = createDataset()
    dataset = dataset.shuffle(buffer_size= 50)
    dataset = dataset.batch(batchSize)
    # train_data = train_data.batch(batchSize)
    # create TensorFlow Iterator object
    # iterator = Iterator.from_structure(train_data.output_types,
    #                                train_data.output_shapes)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # create two initialization ops to switch between the datasets
    # training_init_op = iterator.make_initializer(dataset)
    # validation_init_op = iterator.make_initializer(val_data)

    # sess.run(training_init_op)


    for epoch in range(n_epochs):
        
        sess.run(iterator.initializer)

        print(GPUtil.showUtilization())
        while True:
            try:
                train_img, labels = sess.run(next_element)
                # train_img, labels = map(tf.py_func(input_parser, 
                #         [train_img, labels], [tf.float32, tf.float32]), [train_img, labels])

                train_images = map(crop_image,train_img)
                # train_images = np.array(train_images,dtype=np.float32)
                train_labels = map(crop_image,labels)
                train_labels = map(convert_from_color_segmentation, train_labels)
                # train_labels = np.array(train_labels,dtype=np.float32)


                steploss, _ = sess.run([loss,optimizer], feed_dict= {
                        x: train_images,
                        y: train_labels
                    })
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
            print(steploss)

        print ("Epoch ", epoch, " is done. Saving the model ... ")
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)









