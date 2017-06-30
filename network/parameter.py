import sys


USER = 'usr'

CROP_AREA = (498, 60, 498 + 1000, 60 + 1000)

TRAIN_TEXT = [
    'train.txt',
]

TEST_TEXT = [
    'test.txt',
]

SINGLE_TEST_TEXT = [
    'test.txt',
]


DATA_TRAIN_TEXT_PATH = ['/home/' + USER + '/type/your/path/' + text for text in TRAIN_TEXT]
DATA_TEST_TEXT_PATH = ['/home/' + USER + '/type/your/path/' + text for text in TEST_TEXT]
DATA_SINGLE_TEST_TEXT_PATH = ['/home/' + USER + '/type/your/path/text/' + text for text in SINGLE_TEST_TEXT]

DATA_FILE_PATH = '/home/' + USER + '/type/your/path/'
# DATA_FILE_PATH = '/media/' + USER + '/type/your/path/'

RESTORE_MODE = True
RESTORE_MODEL = 'model'
RESTORE_MODEL_PATH = './models/cnn/' + RESTORE_MODEL

RETRAIN_PATH = 'retrain'

SUMMARIES_PATH = './tensorboard/cnn/'

NUM_OUTPUTS = 6

IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128
CHANNEL_SCALE = 3

DATA_SEQUENCE = [
    [0, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-2, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-4, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-6, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-8, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-10, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-12, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-14, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-16, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-18, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-20, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-22, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-24, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-26, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-28, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
    [-30, (IMAGE_SIZE_X, IMAGE_SIZE_Y)],
]

DATA_SEQUENCE_LENGTH = len(DATA_SEQUENCE)

DATA_SEQUENCE_HEAD = 30
DATA_SEQUENCE_TAIL = 60

RANDOM_PICK_NUM = 100

MAX_STEPS = 2000
BATCH_SIZE = 100
LEARNING_RATE = 1e-5
SAVE_INTERVAL = 20


################################################################################
# liquidate
DATA_SEQUENCE.sort()
DATA_SEQUENCE.reverse()

if DATA_SEQUENCE_TAIL < DATA_SEQUENCE_HEAD:
    print('PAR error: DATA_SEQUENCE_TAIL[%d] < DATA_SEQUENCE_HEAD[%d]'
          % (DATA_SEQUENCE_HEAD, DATA_SEQUENCE_TAIL))
    sys.exit(-1)

################################################################################
