import time
from datetime import datetime
import numpy as np
import random
from PIL import Image
import tensorflow as tf

from network import parameter
from network import cnn3_2d_lstm as nn


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', parameter.TRAIN_TEXT, 'File name of train data.')
flags.DEFINE_string('test', parameter.TEST_TEXT, 'File name of test data.')
flags.DEFINE_string('train_dir', parameter.DATA_FILE_PATH, 'Directory to put the train data.')
flags.DEFINE_integer('max_steps', parameter.MAX_STEPS, 'Number of steps to run training')
flags.DEFINE_integer('batch_size', parameter.BATCH_SIZE, 'Batch size must divide evenly into the dataset sizes')
flags.DEFINE_float('learning_rate', parameter.LEARNING_RATE, 'Initialize learning rate')


def loss(logits_, labels):
    """
    calculate loss
    :param logits_: tensor of logits, float - [batch_size, NUM_CLASSES]
    :param labels: tensor of label, int - [batch_size, NUM_CLASSES]
    :return: cross_entropy
    """

    # calculate cross entropy
    # cross_entropy = - tf.reduce_sum(labels * tf.log(logits_))
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=labels)
    cross_entropy = tf.reduce_mean(diff)

    # TensorBoard
    tf.summary.scalar('cross_entropy', cross_entropy)

    return cross_entropy


def train(loss_, learning_rate):
    """
    define training op
    :param loss_: tensor of loss, result fron loss()
    :param learning_rate: _
    :return: train_step: training op
    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_)
    return train_step


def accuracy(logits_, labels):
    """
    compute accuracy
    :param logits_: result of inference()
    :param labels: tensor of labels, int - [batch_size, NUM_CLASSES]
    :return: accuracy: _
    """

    correct_prediction = tf.equal(tf.argmax(logits_, 1), tf.argmax(labels, 1))
    acc_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', acc_)

    return acc_


def error(logits_, labels):
    """
    a function calculating the error
    :param logits_: result from inference()
    :param labels: tensor of label, int32 - [batch_size, NUM_OUTPUTS]
    :return error: float
    """

    err_ = tf.reduce_mean(tf.square(logits_ - labels))
    tf.summary.scalar('error', err_)

    return err_


if __name__ == '__main__':
    random.seed(1)
    train_images = list()
    train_labels = list()
    test_images = list()
    test_labels = list()

    # data_text = [open(text_path, 'r') for text_path in parameter.DATA_TRAIN_TEXT_PATH]
    data_train_files = [[open(text_path, 'r'), parameter.DATA_FILE_PATH] for text_path in parameter.DATA_TRAIN_TEXT_PATH]
    data_test_files = [[open(text_path, 'r'), parameter.DATA_FILE_PATH] for text_path in parameter.DATA_TEST_TEXT_PATH]

    beginning_time = time.time()

    ####################
    print('SYSTEM: data file path:', parameter.DATA_FILE_PATH)
    print('SYSTEM: data sequence[' + str(len(parameter.DATA_SEQUENCE)) + ']' + ':', parameter.DATA_SEQUENCE)

    with open('log.txt', 'at') as log:
        log.write('########################################\n')
        log.write('run_cnn3_2d_lstm.py\n')
        log.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S\n"))

    for data_file in data_train_files:
        elapsed_time = time.time() - beginning_time

        lines = data_file[0].readlines()
        random.shuffle(lines)

        for line in lines:
            line = line.rstrip()
            element = line.split()
            print('SYSTEM:', element)

            for i in range(parameter.DATA_SEQUENCE_HEAD, parameter.DATA_SEQUENCE_TAIL):
                image_list = list()
                print('SYSTEM: --%02d' % i)

                for j, frame in enumerate(parameter.DATA_SEQUENCE):
                    image_path = element[0] + ('%06d' % (int(element[1]) + frame[0] + i)) + '.jpg'
                    image_path = data_file[1] + image_path

                    img = Image.open(image_path, 'r')

                    crop_img = img.crop(parameter.CROP_AREA)
                    resize_img = crop_img.resize(frame[1])
                    resize_img = np.array(resize_img)
                    image_list.append(resize_img.astype(np.float32) / 255.0)

                train_images.append(image_list)
                tmp = np.zeros(parameter.NUM_OUTPUTS)
                tmp[int(element[2])] = 1
                train_labels.append(tmp)

        print('SYSTEM: read time:{0}[s]'.format(elapsed_time))
        data_file[0].close()

    for data_file in data_test_files:
        elapsed_time = time.time() - beginning_time

        for line in data_file[0]:
            line = line.rstrip()
            element = line.split()
            print('SYSTEM:', element)

            for i in range(parameter.DATA_SEQUENCE_HEAD, parameter.DATA_SEQUENCE_TAIL):
                image_list = list()
                print('SYSTEM: --%02d' % i)

                for j, frame in enumerate(parameter.DATA_SEQUENCE):
                    image_path = element[0] + ('%06d' % (int(element[1]) + frame[0] + i)) + '.jpg'
                    image_path = data_file[1] + image_path

                    img = Image.open(image_path, 'r')

                    crop_img = img.crop(parameter.CROP_AREA)
                    resize_img = crop_img.resize(frame[1])
                    resize_img = np.array(resize_img)
                    image_list.append(resize_img.astype(np.float32) / 255.0)

                test_images.append(image_list)
                tmp = np.zeros(parameter.NUM_OUTPUTS)
                tmp[int(element[2])] = 1
                test_labels.append(tmp)

        print('SYSTEM: read time:{0}[s]'.format(elapsed_time))
        data_file[0].close()

    print('SYSTEM: train images:', len(train_labels))
    print('SYSTEM: test images:', len(test_labels))
    ####################

    ####################
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(None,
                                               parameter.DATA_SEQUENCE_LENGTH,
                                               parameter.IMAGE_SIZE_X,
                                               parameter.IMAGE_SIZE_Y,
                                               parameter.CHANNEL_SCALE))
    labels_placeholder = tf.placeholder(tf.float32,
                                        shape=(None, parameter.NUM_OUTPUTS))
    keep_prob = tf.placeholder(tf.float32)
    lstm_memory_size = tf.placeholder(tf.int32)
    ####################

    ####################
    logits = nn.inference(images_placeholder, keep_prob, lstm_memory_size)
    loss_val = loss(logits, labels_placeholder)
    train_op = train(loss_val, parameter.LEARNING_RATE)
    acc = accuracy(logits, labels_placeholder)
    err = error(logits, labels_placeholder)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(parameter.SUMMARIES_PATH + 'train/', sess.graph)
        test_writer = tf.summary.FileWriter(parameter.SUMMARIES_PATH + 'test/', sess.graph)

        beginning_time = time.time()

        for step in range(parameter.MAX_STEPS):
            if step % parameter.SAVE_INTERVAL == 0 and step != 0:
                save_path = saver.save(sess, parameter.RESTORE_MODEL_PATH, global_step=step)

            for i in range(int(len(train_labels) / parameter.BATCH_SIZE)):
                batch_offset = parameter.BATCH_SIZE * i

                sess.run(train_op, feed_dict={
                    images_placeholder: train_images[batch_offset:batch_offset + parameter.BATCH_SIZE],
                    labels_placeholder: train_labels[batch_offset:batch_offset + parameter.BATCH_SIZE],
                    keep_prob: 0.5,
                    lstm_memory_size: len(train_labels[batch_offset:batch_offset + parameter.BATCH_SIZE])
                })

            random_train_images = list()
            random_train_labels = list()
            random_test_images = list()
            random_test_labels = list()

            for num in range(0, parameter.RANDOM_PICK_NUM):
                random_train_index = random.randint(0, len(train_labels) - 1)
                random_train_images.append(train_images[random_train_index])
                random_train_labels.append(train_labels[random_train_index])

                random_test_index = random.randint(0, len(test_labels) - 1)
                random_test_images.append(test_images[random_test_index])
                random_test_labels.append(test_labels[random_test_index])

            train_summary, _ = sess.run([merged, train_op], feed_dict={
                images_placeholder: random_train_images,
                labels_placeholder: random_train_labels,
                keep_prob: 1.0,
                lstm_memory_size: len(random_train_labels)
            })

            test_summary, test_acc = sess.run([merged, acc], feed_dict={
                images_placeholder: random_test_images,
                labels_placeholder: random_test_labels,
                keep_prob: 1.0,
                lstm_memory_size: len(random_test_labels)
            })

            train_writer.add_summary(train_summary, step)
            test_writer.add_summary(test_summary, step)

            print('step %d, test acc %g' % (step, test_acc))

        save_path = saver.save(sess, parameter.RESTORE_MODEL_PATH)
        print('SYSTEM: last model at', save_path)

    elapsed_time = time.time() - beginning_time
    print('elapsed time: {0}[s]'.format(elapsed_time))
    ####################
