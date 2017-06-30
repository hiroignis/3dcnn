import numpy as np
from PIL import Image
import tensorflow as tf

from network import parameter
from network import cnn2d as nn


if __name__ == '__main__':
    test_images = list()
    test_labels = list()

    data_text = open(parameter.DATA_TRAIN_TEXT_PATH, 'r')
    data_file1 = [data_text, parameter.DATA_FILE_PATH]

    data_files = [data_file1]

    ####################
    num_samples = 0
    print('SYSTEM: data file path:', parameter.DATA_FILE_PATH)
    print('SYSTEM: data sequence[' + str(len(parameter.DATA_SEQUENCE)) + ']' + ':', parameter.DATA_SEQUENCE)
    for data_file in data_files:

        for line in data_file[0]:
            line = line.rstrip()
            element = line.split()
            print('SYSTEM:', element)

            for i in range(parameter.DATA_SEQUENCE_HEAD, parameter.DATA_SEQUENCE_TAIL):
                image_list = list()
                print('SYSTEM: --%02d' % i)

                for j, frame in enumerate(parameter.DATA_SEQUENCE):
                    image_path = element[0]+ ('%06d' % (int(element[1]) + frame[0] + i)) + '.jpg'
                    image_path = data_file[1] + image_path
                    print('SYSTEM: ----%s' % image_path)

                    img = Image.open(image_path, 'r')
                    resize_img = img.resize(frame[1])
                    resize_img = np.array(resize_img)
                    image_list.append(resize_img.astype(np.float32) / 255.0)

                test_images.append(image_list)
                tmp = np.zeros(parameter.NUM_OUTPUTS)
                tmp[int(element[2])] = 1
                test_labels.append(tmp)

            num_samples += 1

        data_file[0].close()

    print('SYSTEM: test images:', len(test_labels))
    ####################

    ####################
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(None,
                                               parameter.IMAGE_SIZE_X,
                                               parameter.IMAGE_SIZE_Y,
                                               parameter.CHANNEL_SCALE))
    labels_placeholder = tf.placeholder(tf.float32,
                                        shape=(None, parameter.NUM_OUTPUTS))
    keep_prob = tf.placeholder(tf.float32)
    ####################

    logits = nn.inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()

    # new_saver = tf.train.import_meta_graph('./sample/model.meta')

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, '/home/ignis/IGNIS/PycharmProjects/otmove/models/cnn/model-40')
    saver.restore(sess, parameter.RESTORE_MODEL_PATH)
    # saver.restore(sess, './sample/model')

    n = [0, 0, 0, 0, 0, 0]

    for i in range(0, len(test_images)):
        # pred = np.argmax(logits.eval(feed_dict={
        #     images_placeholder: test_images[i],
        #     keep_prob: 1.0})[0])

        pred = logits.eval(feed_dict={
            images_placeholder: test_images[i],
            keep_prob: 1.0})[0]

        decision = np.argmax(pred)

        # print(test_labels[i], pred)
        for j, k in enumerate(test_labels[i]):
            if k == 1:
                if j != decision:
                    # print(test_labels[i], pred)
                    print(j, decision)
                    n[j] += 1

    print(n, '/', len(test_images))
