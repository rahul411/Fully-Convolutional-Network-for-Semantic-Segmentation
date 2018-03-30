import numpy as np
from tensorflow.python.client import device_lib

num_classes = 21
img_rows = 224
img_cols =224

def computeIoU(y_pred_batch, y_true_batch):
    v = np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))]))
    # return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))])) 
    print(v)
    return v

def pixelAccuracy(y_pred, y_true):
    # y_pred = np.argmax(np.reshape(y_pred,[img_rows,img_cols, num_classes]),axis=2)
    # y_true = np.argmax(np.reshape(y_true,[img_rows,img_cols, num_classes]),axis=2)
    # y_pred = np.argmax(y_pred,axis=2)
    y_true = np.argmax(y_true,axis=2)
    # print(y_true)
    y_pred = y_pred * (y_true>0)

    return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  (np.sum(y_true>0)+1) 

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']