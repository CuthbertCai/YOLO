from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import time
import sys
from datetime import datetime

from yolo.solver.solver import Solver

class YoloSolver(Solver):
    """Yolo solver

    """

    def __init__(self, dataset, net, common_params, solver_params):
        #process params
        self.moment = float(solver_params['moment'])
        self.learning_rate = float(solver_params['learning_rate'])
        self.batch_size = int(common_params['batch_size'])
        self.height = int(common_params['image_size'])
        self.width = int(common_params['image_size'])
        self.max_objects = int(common_params['max_objects_per_image'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])
        self.dataset = dataset
        self.net = net
#        self.construct_graph()

    def _train(self):
        """Train model
        Create an optimizer and apply to all trainable variable
        :return: train_op: op for training
        """
        opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
        grads = opt.compute_gradients(self.total_loss)
        apply_grads = opt.apply_gradients(grads, global_step= self.global_step)
        with tf.control_dependencies([apply_grads]):
            train_op = tf.no_op(name='train')
        return  train_op

#    def construct_graph(self):
#       #construct graph
#       self.global_step = tf.Variable(0, trainable= False)
#       self.images = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, 3])
#       self.labels = tf.placeholder(tf.float32,[self.batch_size, self.max_objects, 5])
#       self.objects_num = tf.placeholder(tf.int32, [self.batch_size])

#       self.predicts = self.net.inference(self.images)

#       self.total_loss, self.predictor = self.net.loss(self.predicts, self.labels, self.objects_num)
#       tf.summary.scalar('total_losses', self.total_loss)
#       self.train_op = self._train()

    def solve(self):
        with tf.Graph().as_default():
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.images, self.labels, self.objects_num = self.dataset.batch()
            self.predicts = self.net.inference(self.images)
            self.total_loss, self.predictor = self.net.loss(self.predicts, self.labels, self.objects_num)
            tf.summary.scalar('total_losses', self.total_loss)
            self.train_op = self._train()
            saver1 = tf.train.Saver(self.net.pretrain_collection, write_version= tf.train.SaverDef.V2)
            saver2 = tf.train.Saver(self.net.trainable_collection, write_version= tf.train.SaverDef.V2)

            init = tf.global_variables_initializer()

            summary_op = tf.summary.merge_all()

            sess = tf.Session()

            sess.run(init)
#            saver1 = tf.train.import_meta_graph(self.train_dir + '/model.ckpt-5000.meta')
            saver1.restore(sess, self.train_dir + '/model.ckpt-5000')
            #saver2.restore(sess, tf.train.latest_checkpoint(self.train_dir))
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            tf.train.start_queue_runners(sess)


            for step in range(self.max_iterators):
                start_time = time.time()

                #_, losses, predictor = sess.run([self.train_op, self.total_loss, self.predictor],
                #                           feed_dict={self.images:batch_images, self.labels:batch_labels, self.objects_num:batch_objects_num})
                _, losses_value, predictor = sess.run([self.train_op, self.total_loss, self.predictor])
                duration = time.time() - start_time

                assert not np.isnan(losses_value), 'Model diverged with loss = NaN'

                if step % 1 == 0:
                    num_examples_per_step = self.dataset.batch_size
                    examples_per_sec = float(num_examples_per_step) / duration
                    sec_per_batch = float(duration)

                    print('%s: step %d, losses = %.5f (%.1f examples/sec; %.3f sec/batch' % (datetime.now(), step, losses_value, examples_per_sec, sec_per_batch))

                    sys.stdout.flush()

                if step % 100 == 0:
#                   summary_str = sess.run(summary_op, feed_dict={self.images:batch_images, self.labels:batch_labels,
#                                                                  self.objects_num:batch_objects_num})
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                if step % 5000 == 0:
                    saver2.save(sess, self.train_dir + '/model.ckpt', global_step= step)
            sess.close()

