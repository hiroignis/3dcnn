import numpy as np
from PIL import Image
import tensorflow as tf

from neteork import ignis_parameter as parameter
from network import cnn3d as nn


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == '__main__':
    """
    Same as test_cnn3_2d.py
    """
    test_images = list()
    test_labels = list()

    data_test_files = [[open(text_path, 'r'), parameter.DATA_FILE_PATH] for text_path in parameter.DATA_SINGLE_TEST_TEXT_PATH]

    ####################
    print('SYSTEM: data file path:', parameter.DATA_FILE_PATH)
    print('SYSTEM: data sequence[' + str(len(parameter.DATA_SEQUENCE)) + ']' + ':', parameter.DATA_SEQUENCE)

    for data_file in data_test_files:

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
                    print('SYSTEM: ----%s' % image_path)

                    img = Image.open(image_path, 'r')

                    crop_img = img.crop(parameter.CROP_AREA)
                    resize_img = crop_img.resize(frame[1])
                    resize_img = np.array(resize_img)
                    image_list.append(resize_img.astype(np.float32) / 255.0)

                test_images.append(image_list)
                tmp = np.zeros(parameter.NUM_OUTPUTS)
                tmp[int(element[2])] = 1
                test_labels.append(tmp)

        data_file[0].close()

    print('SYSTEM: test images:', len(test_labels))
    ####################

    ####################
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(None,
                                               len(parameter.DATA_SEQUENCE),
                                               parameter.IMAGE_SIZE_X,
                                               parameter.IMAGE_SIZE_Y,
                                               parameter.CHANNEL_SCALE))
    labels_placeholder = tf.placeholder(tf.float32,
                                        shape=(None, parameter.NUM_OUTPUTS))
    keep_prob = tf.placeholder(tf.float32)
    ####################

    logits = nn.inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print('SYSTEM: model path:', parameter.RESTORE_MODEL_PATH)
    saver.restore(sess, parameter.RESTORE_MODEL_PATH)

    seq_predict = list()
    sum_predict = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
    sum_decision = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

    for t, (test_image, test_label) in enumerate(zip(test_images, test_labels)):
        predict = logits.eval(feed_dict={
            images_placeholder: [test_image],
            keep_prob: 1.0})[0]

        predict = np.array(predict)
        predict = softmax(predict)

        sum_predict += predict

        decision = np.argmax(predict)
        sum_decision[decision] += 1

        for j, k in enumerate(test_label):
            if k == 1:
                seq_predict.append([predict, decision, j])
                break

    avg_predict = sum_predict / len(test_images)
    avg_decision = sum_decision / len(test_images)

    with open('seq_predict.csv', 'xt') as f:
        for test_text in parameter.DATA_SINGLE_TEST_TEXT_PATH:
            f.write(str(test_text) + ', ')
        f.write('-1\n')

        f.write('frame, predict0, predict1, predict2, predict3, predict4, predict5, decision, ans' + '\n')
        for i, val, in enumerate(seq_predict):
            f.write(str(i + parameter.DATA_SEQUENCE_HEAD))

            for pred in val[0]:
                f.write(', ' + str(pred))
            f.write(', ' + str(val[1])
                    + ', ' + str(val[2]) + '\n')

    with open('avg_predict.csv', 'xt') as f:
        for test_text in parameter.DATA_SINGLE_TEST_TEXT_PATH:
            f.write(str(test_text) + ', ')
        f.write('-1\n')

        for i, val in enumerate(avg_predict):
            f.write(str(i) + ', ' + str(val) + '\n')

    with open('avg_decision.csv', 'xt') as f:
        for test_text in parameter.DATA_SINGLE_TEST_TEXT_PATH:
            f.write(str(test_text) + ', ')
        f.write('-1\n')

        for i, val in enumerate(avg_decision):
            f.write(str(i) + ', ' + str(val) + '\n')
