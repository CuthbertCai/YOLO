from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import re

from yolo.net.net import Net


class YoloNet(Net):
    """Class of YOLO Net

    """

    def __init__(self, common_params, net_params, test=False):
        """

        :param common_params: a params dict
        :param net_params: a params dict
        """
#        self.pretrain_collection = []
#        self.trainable_collection = []
        super(YoloNet, self).__init__(common_params, net_params)
        # process params
        self.image_size = int(common_params['image_size'])
        self.num_classes = int(common_params['num_classes'])
        self.cell_size = int(net_params['cell_size'])
        self.boxes_per_cell = int(net_params['boxes_per_cell'])
        self.batch_size = int(common_params['batch_size'])
        self.weight_decay = float(net_params['weight_decay'])

        if not test:
            self.object_scale = float(net_params['object_scale'])
            self.noobject_scale = float(net_params['noobject_scale'])
            self.class_scale = float(net_params['class_scale'])
            self.coord_scale = float(net_params['coord_scale'])

    def inference(self, images):
        """Build the YOLO model

        :param images: 4-D Tensor [batch_size, image_height, image_width, channels]
        :return: predicts: 4-D Tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell
        """
        net = self.conv2d('conv1', images, [7, 7, 3, 64], stride=2)
        net = self.max_pool(net, [2, 2], 2)
        net = self.conv2d('conv2', net, [3, 3, 64, 192])
        net = self.max_pool(net, [2, 2], 2)
        net = self.conv2d('conv3', net, [1, 1, 192, 128])
        net = self.conv2d('conv4', net, [3, 3, 128, 256])
        net = self.conv2d('conv5', net, [1, 1, 256, 256])
        net = self.conv2d('conv6', net, [3, 3, 256, 512])
        net = self.max_pool(net, [2, 2], 2)

        conv_num = 7
        for i in range(4):
            net = self.conv2d('conv' + str(conv_num), net, [1, 1, 512, 256])
            conv_num += 1

            net = self.conv2d('conv' + str(conv_num), net, [3, 3, 256, 512])
            conv_num += 1

        net = self.conv2d('conv' + str(conv_num), net, [1, 1, 512, 512])
        conv_num += 1

        net = self.conv2d('conv' + str(conv_num), net, [3, 3, 512, 1024])
        conv_num += 1
        net = self.max_pool(net, [2, 2], 2)

        for i in range(2):
            net = self.conv2d('conv' + str(conv_num), net, [1, 1, 1024, 512])
            conv_num += 1

            net = self.conv2d('conv' + str(conv_num), net, [3, 3, 512, 1024])
            conv_num += 1

        net = self.conv2d('conv' + str(conv_num), net, [3, 3, 1024, 1024])
        conv_num += 1

        net = self.conv2d('conv' + str(conv_num), net, [3, 3, 1024, 1024], stride=2)
        conv_num += 1

        net = self.conv2d('conv' + str(conv_num), net, [3, 3, 1024, 1024])
        conv_num += 1

        net = self.conv2d('conv' + str(conv_num), net, [3, 3, 1024, 1024])
        conv_num += 1

        # Fully connected layers
        net = self.local('local1', net, 49 * 1024, 4096)

        net = tf.nn.dropout(net, keep_prob=0.5)

        net = self.local('local2', net, 4096, self.cell_size * self.cell_size * (self.num_classes
                                                                                 + 5 * self.boxes_per_cell))

        net = tf.reshape(net, [tf.shape(net)[0], self.cell_size, self.cell_size, self.num_classes
                               + 5 * self.boxes_per_cell])

        predicts = net

        return predicts

    def iou(self, boxes1, boxes2):
        """Calculate the ious

        :param boxes1: 4-D Tensor [cell_size, cell_size, boxes_per_cell, 4]  ====> [x_center, y_center, w, h]
        :param boxes2: 1-D Tensor [4] =====> [x_center, y_center, w, h]
        :return: 3-D Tensor [cell_size, cell_size, boxes_per_cell]
        """
        boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                           boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2],
                          axis=3)
        boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                           boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

        # calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        # intersection
        intersection = rd - lu

        inter_area = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

        inter_area = mask * inter_area

        # calculate the area of boxes1 and boxes2
        area_1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        area_2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        return inter_area / (area_1 + area_2 - inter_area + 1e-6)

    def cond1(self, num, object_num, loss, predict, label, predictor):
        """if num < object_num

        :return: num < object_num
        """
        return num < object_num

    def cal_loss(self, num, object_num, loss, predict, labels, predictor):
        """Calculate loss

        :param predict: predicted 3-D Tensor [cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        :param labels: [max_objects, 5] [x_center, y_center, w, h, class]
        """
        label = labels[num, :]
        label = tf.reshape(label, [-1])

        # calculate objects tensor [cell_size, cell_size], reshape the object in image_size into cell_size
        min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
        max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)

        min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
        max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)

        max_x = tf.ceil(max_x)
        max_y = tf.ceil(max_y)

        shape = tf.cast([max_y - min_y, max_x - min_x], dtype=tf.int32)
        objects = tf.ones(shape, tf.float32)

        shape = tf.cast([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x], tf.int32)
        shape = tf.reshape(shape, [2, 2])
        objects = tf.pad(objects, shape)

        # calculate objects tensor [cell_size, cell_size]
        # calculate responsible tensor [cell_size, cell_size]
        center_x = label[0] / (self.image_size / self.cell_size)
        center_y = label[1] / (self.image_size / self.cell_size)
        center_x = tf.floor(center_x)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        shape = tf.cast([center_y, self.cell_size - center_y - 1, center_x, self.cell_size - center_x - 1],
                        tf.int32)
        shape = tf.reshape(shape, [2, 2])
        response = tf.pad(response, shape)

        # calculate iou_predict_truth [cell_size, cell_size, boxes_per_cell]

        predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]

        predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size,
                                         self.image_size, self.image_size]

        base_boxes = np.zeros([self.cell_size, self.cell_size, 4], dtype=np.float32)

        for y in range(self.cell_size):
            for x in range(self.cell_size):
                base_boxes[y, x, :] = [(self.image_size / self.cell_size * x),
                                       (self.image_size / self.cell_size * y), 0, 0]
        base_boxes = tf.tile(tf.reshape(base_boxes, [self.cell_size, self.cell_size, 1, 4]),
                             [1, 1, self.boxes_per_cell, 1])
        base_boxes = tf.convert_to_tensor(base_boxes, dtype=tf.float32)

        predict_boxes = predict_boxes + base_boxes
        iou_predict_truth = self.iou(predict_boxes, label[0:4])

        # calculate C [cell_size, cell_size, boxes_per_cell], choose the grid cell which contain the object
        C = iou_predict_truth * tf.reshape(response, shape=[self.cell_size, self.cell_size, 1])

        # calculate I [cell_size, cell_size, boxes_per_cell], choose which bounding box in the cell contain the object
        I = iou_predict_truth * tf.reshape(response, shape=[self.cell_size, self.cell_size, 1])

        max_I = tf.reduce_max(I, axis=2, keep_dims=True)
        I = tf.cast((I >= max_I), dtype=tf.float32) * tf.reshape(response, [self.cell_size, self.cell_size, 1])

        # calculate no_I [cell_size, cell_size, boxes_per)cell], all the bounding boxes that don't contain the object
        no_I = tf.ones_like(I, dtype=tf.float32) - I

        # calculate the confidence that the bounding boxes contain the objects
        p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

        # calculate truth x, y, sqrt_w, sqrt_h
        x = label[0]
        y = label[1]
        sqrt_w = tf.sqrt(label[2])
        sqrt_h = tf.sqrt(label[3])

        # calculate the predicted x, y, sqrt_w, sqrt_h
        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]
        p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

        # calculate the class probabilities of the grid cells containing objects
        P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)
        P = P * tf.ones([self.cell_size, self.cell_size, self.num_classes])

        # calculate the predicted class probabilities
        p_P = predict[:, :, 0:self.num_classes]

        # class loss
        class_loss = tf.nn.l2_loss(tf.reshape(objects,
                                              [self.cell_size, self.cell_size, 1]) * (p_P - P)) * self.class_scale

        # no object loss
        no_obj_loss = tf.nn.l2_loss(no_I * p_C) * self.noobject_scale

        # object loss
        obj_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale

        # coordnate loss
        coord_loss = (tf.nn.l2_loss(I * ((x - p_x) / (self.image_size / self.cell_size))) +
                      tf.nn.l2_loss(I * ((y - p_y) / (self.image_size / self.cell_size))) +
                      tf.nn.l2_loss(I * ((sqrt_h - p_sqrt_h) / (self.image_size / self.cell_size))) +
                      tf.nn.l2_loss(I * ((sqrt_w - p_sqrt_w) / (self.image_size / self.cell_size)))) * self.coord_scale

        predictor = I

        return num + 1, object_num, [loss[0] + class_loss, loss[1] + obj_loss, loss[2] + no_obj_loss,
                                     loss[3] + coord_loss], \
               predict, labels, predictor

    def loss(self, predicts, labels, objects_num):
        """Add loss to all trainable variable

        :param predicts: 4-D Tensor [batch_size, cell_size, cell_size, num_classes + boxes_per_cell * 5]
        :param labels: 3-D Tensor [batch_size, max_objects, 5]
        :param objects_num: 1-D Tensor [batch_size]
        """
        class_loss = tf.constant(0, tf.float32)
        obj_loss = tf.constant(0, tf.float32)
        no_obj_loss = tf.constant(0, tf.float32)
        coord_loss = tf.constant(0, tf.float32)
        predictor = tf.ones([self.cell_size, self.cell_size, self.boxes_per_cell])
        loss = [0, 0, 0, 0]

        for i in range(self.batch_size):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            object_num = objects_num[i]
            tuple_results = tf.while_loop(self.cond1, self.cal_loss, [tf.constant(0), object_num,
                                                                      [class_loss, obj_loss, no_obj_loss, coord_loss],
                                                                      predict, label, predictor])

            for j in range(4):
                loss[j] = tuple_results[2][j]

            predictor = tuple_results[5]

        tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)

        tf.summary.scalar('class_loss', loss[0] / self.batch_size)
        tf.summary.scalar('obj_loss', loss[1] / self.batch_size)
        tf.summary.scalar('no_obj_loss', loss[2] / self.batch_size)
        tf.summary.scalar('coord_loss', loss[3] / self.batch_size)

        tf.summary.scalar('weight_decay', tf.add_n(tf.get_collection('losses')) - loss[0] - loss[1] - loss[2] - loss[3]
                          / self.batch_size)

        return tf.add_n(tf.get_collection('losses'), name='total_losses'), predictor
