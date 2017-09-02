import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf
import cv2
import numpy as np


classes_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'preson', 'potteplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def process_predict(predicts, common_params, net_params):
    p_classes = predicts[0, :, :, 0:common_params['num_classes']]
    C = predicts[0, :, :, common_params['num_classes']:common_params['num_classes'] + net_params['boxes_per_cell']]
    coordinate = predicts[0, :, :,common_params['num_classes'] + net_params['boxes_per_cell']:]

    #C_max = tf.reduce_max(C, axis= 2, keep_dims= True)
    #C_temp = tf.cast((C >= C_max), dtype= tf.float32)
    #C = C * C_temp

    p_classes = np.reshape(p_classes, [net_params['cell_size'], net_params['cell_size'], 1, common_params['num_classes']])
    C = np.reshape(C, [net_params['cell_size'], net_params['cell_size'], net_params['boxes_per_cell'], 1])

    p = C * p_classes

    index = np.argmax(p)
    index = np.unravel_index(index, p.shape)

    class_num = index[3]

    coordinate = np.reshape(coordinate, [net_params['cell_size'], net_params['cell_size'],
                                                net_params['boxes_per_cell'], 4])

    max_coordinate = coordinate[index[0], index[1], index[2], :]

    center_x = max_coordinate[0]
    center_y = max_coordinate[1]
    w = max_coordinate[2]
    h = max_coordinate[3]

    center_x = (index[1] + center_x) * (common_params['image_size'] / net_params['cell_size'])
    center_y = (index[2] + center_y) * (common_params['image_size'] / net_params['cell_size'])
    w = w * common_params['image_size']
    h = h * common_params['image_size']

    xmin = center_x - w / 2.0
    ymin = center_y - h / 2.0
    xmax = center_x + w / 2.0
    ymax = center_y + h / 2.0

    return xmin, ymin, xmax, ymax, class_num

common_params = {'image_size': 448, 'num_classes': 20, 'batch_size': 1}
net_params = {'cell_size': 7, 'boxes_per_cell': 2, 'weight_decay': 5e-4}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (common_params['batch_size'], common_params['image_size'],
                                    common_params['image_size'], 3))

predicts = net.inference(image)

sess = tf.Session()

test_img = cv2.imread('cat.jpg')
resized_img = cv2.resize(test_img, (common_params['image_size'], common_params['image_size']))

test_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

test_img = test_img.astype(np.float32)
test_img = test_img / 255.0 * 2 - 1
test_img = np.reshape(test_img, [common_params['batch_size'], common_params['image_size'],
                                 common_params['image_size'], 3])

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess, '/home/cuthbert/Program/YOLO/models/pretrain/yolo_tiny.ckpt')

test_predict = sess.run(predicts, feed_dict={image: test_img})

xmin, ymin, xmax, ymax, class_num = process_predict(test_predict, common_params, net_params)
class_name = classes_name[class_num]

cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
cv2.putText(resized_img, class_name, (int(xmax), int(ymax)), 2, 0.5, (0, 0, 255))
cv2.imwrite('cat_out.jpg', resized_img)
sess.close()
