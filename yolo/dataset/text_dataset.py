import random
import cv2
import numpy as np
from queue import Queue
from yolo.dataset.dataset import DataSet
import tensorflow as tf
from threading import Thread
class TextDataSet(DataSet):
    """TextDataSet
    process text input file dataset
     text file format: image path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
    """
    def __init__(self, common_params, dataset_params):
        """

        :param common_params: A params dict
        :param dataset_params: A params dict
        """
        #process params
        self.data_path = str(dataset_params['path'])
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.thread_num = int(dataset_params['thread_num'])
        self.max_objects = int(common_params['max_objects_per_image'])
        self.min_fraction_of_examples_in_queue = float(dataset_params['min_fraction'])
        #record and image_label queue
        self.record_queue = Queue(maxsize= 10000)
        #self.image_label_queue = Queue(maxsize= 512)

        self.record_list = []

        #filling the record list
        input_file = open(self.data_path, 'r')

        for line in input_file:
            line = line.strip()
            ss = line.split(' ')
#            ss[1:] = [num for num in ss[1:]]
            self.record_list.append(ss)

        self.record_point = 0
        self.record_number = len(self.record_list)

        self.num_batch_per_epoch = int(self.record_number / self.batch_size)



    def record_producer(self):
        """record_queue's processor

        """
        while True:
            if self.record_point % self.record_number ==0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point +=1
            # print('record_point %d' %self.record_point)

    def record_process(self, record):
        """record process

        :return:
                images: 3-D Tensor
                labels: 2-D Tensor [self.max_objects, 5] =====> (center_x, center_y, w, h, class_num]
                object_num: total objectt num
        """
        image = cv2.imread(record[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]

        width_rate = self.width * 1.0 / w
        height_rate = self.height * 1.0 / h

        image = cv2.resize(image, (self.height, self.width))

        labels = [[0,0,0,0,0]] * self.max_objects

        i = 1
        object_num = 0
        while i < len(record):
            xmin = float(record[i])
            ymin = float(record[i + 1])
            xmax = float(record[i + 2])
            ymax = float(record[i + 3])
            class_num = int(record[i + 4])

            center_x = (xmin + xmax) * width_rate / 2
            center_y = (ymin + ymax) * height_rate / 2

            box_w = (xmax - xmin) * width_rate
            box_h = (ymax - ymin) * height_rate

            labels[object_num] = [center_x, center_y, box_w, box_h, class_num]
            object_num += 1
            i += 5
            if object_num >= self.max_objects:
                break

        return [image, labels, object_num]

#    def record_customer(self):
#        """record queue's customer
#
#        """
#        while True:
#            item = self.record_queue.get()
#            out = self.record_process(item)
#            self.image_label_queue.put(out)
#            print('record_customer')

    def batch(self):
        """get batch

        :return:
            images : 4-D Tensor [batch_size, heigth, width, 3]
            labels : 3-D Tensor [batch_size, max_objects, 5]
            objects_num : 1-D Tensor [batch_size]
        """
#        t_record_producer = Thread(target= self.record_producer)
#        t_record_producer.daemon = True
#        t_record_producer.start()

#        t_record_customer = []
#        for i in range(self.thread_num):
#            t_record_customer[i] = Thread(target= self.record_customer())
#            print('t_record_customer %d' %i)
#            t_record_customer[i].daemon = True
#            t_record_customer[i].start()

        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()
        image, label, object_num = self.record_process(self.record_queue.get())
        min_queue_examples = int(self.min_fraction_of_examples_in_queue * self.num_batch_per_epoch * self.batch_size)
        images, labels, objects_num = tf.train.shuffle_batch(
            [image, label, object_num],
            batch_size= self.batch_size,
            num_threads= self.thread_num,
            capacity= min_queue_examples + 3 * self.batch_size,
            min_after_dequeue= min_queue_examples
        )

#        images = []
#        labels = []
#        objects_num = []

#        for i in range(self.batch_size):
#            image, label, object_num = self.image_label_queue.get()
#            print('image_label_queue')
#            images.append(image)
#            labels.append(label)
#            objects_num.append(object_num)

#        images = np.asarray(images, dtype=np.float32)
        images = images / 255 * 2 - 1
#        labels = np.asarray(labels, dtype= np.float32)
#        objects_num = np.asarray(objects_num, dtype= np.int32)
        return images, labels, objects_num

