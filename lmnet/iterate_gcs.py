import tracemalloc
import time

import tensorflow as tf

from lmnet.datasets.tfds import *
from lmnet.datasets.dataset_iterator import *


def main():
    tracemalloc.start()

    init_begin = time.time()
    ds = TFDSClassification(tfds_name="imagenet2012", subset="train", tfds_data_dir="gs://lmfs-backend-us-west1/shared/tensorflow_datasets", batch_size=8)
    # ds = TFDSClassification(tfds_name="cifar10", subset="train", tfds_data_dir="/Users/fujiwara/tensorflow_datasets", batch_size=8)
    ds.tf_dataset = ds.tf_dataset.map(
        lambda record: {'image': tf.image.resize(record['image'], [160, 160]), 'label': record['label']},
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dsi = DatasetIterator(ds)
    init_end = time.time()

    load_begin = time.time()
    prev =  time.time()

    for i in range(int(1281167 / 8)):
        if i % 100 == 0:
            now = time.time()
            print("=====")
            print(i, (prev - now) / 100)
            prev = now

            print("=====")
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            for stat in top_stats[:10]:
                print(stat)

        batch = dsi.feed()
    load_end = time.time()

    print('Init: {}'.format(init_end - init_begin))
    print('Load: {}'.format(load_end - load_begin))


if __name__ == '__main__':
    main()
