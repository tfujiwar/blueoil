from easydict import EasyDict
import tensorflow as tf

from lmnet.common import Tasks
from lmnet.datasets.tfds import TFDSClassification
from lmnet.data_processor import Sequence
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    Crop,
    FlipLeftRight,
    Hue,
)
from lmnet.networks.classification.darknet import DarknetQuantize
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
)
from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = DarknetQuantize
DATASET_CLASS = TFDSClassification

IMAGE_SIZE = [224, 224]
BATCH_SIZE = 8
DATA_FORMAT = "NCHW"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 2000000
SAVE_STEPS = 50000
TEST_STEPS = 50000
SUMMARISE_STEPS = 10000

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# distributed training
IS_DISTRIBUTION = False

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255()
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.train.polynomial_decay
# TODO(wakiska): It is same as original yolov2 paper (batch size = 128).
NETWORK.LEARNING_RATE_KWARGS = {"learning_rate": 1e-1, "decay_steps": 1600000, "power": 4.0, "end_learning_rate": 0.0}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2.0,
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}
NETWORK.QUANTIZE_FIRST_CONVOLUTION = True
NETWORK.QUANTIZE_LAST_CONVOLUTION = False

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Crop(size=IMAGE_SIZE, resize=256),
    FlipLeftRight(),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
])
DATASET.TFDS_NAME = "imagenet2012"
DATASET.TFDS_DATA_DIR = "gs://lmfs-backend-us-west1/shared/tensorflow_datasets"
DATASET.TFDS_IMAGE_SIZE = [256, 256]