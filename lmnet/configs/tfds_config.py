from easydict import EasyDict
import tensorflow as tf
import tensorflow_datasets as tfds

from lmnet.common import Tasks
from lmnet.networks.classification.lmnet_v1 import LmnetV1Quantize
from lmnet.datasets.tfds import TFDSClassification

from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
    PerImageStandardization,
)
from lmnet.data_augmentor import (
    FlipTopBottom,
)
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LmnetV1Quantize

DATASET_CLASS = TFDSClassification

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 8
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = None

MAX_EPOCHS = 10
SAVE_STEPS = 1000
TEST_STEPS = 1000
SUMMARISE_STEPS = 100

# distributed training
IS_DISTRIBUTION = False

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255()
])
POST_PROCESSOR = None

NETWORK = EasyDict()

NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {'learning_rate': 0.01}
NETWORK.LEARNING_RATE_FUNC = None
NETWORK.LEARNING_RATE_KWARGS = None

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005

# quantize
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255(),
    PerImageStandardization(),
])
DATASET.AUGMENTOR = Sequence([
    FlipTopBottom(),
])

DATASET.TFDS_NAME = 'cifar10'
DATASET.TFDS_DATA_DIR = 'gs://lmfs-backend-us-west1/shared/tensorflow_datasets'
DATASET.TFDS_DOWNLOAD = True