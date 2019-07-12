# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import pytest
from lmnet.datasets.tfds import TFDSClassification
from lmnet.datasets.dataset_iterator import DatasetIterator

# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_tfds(tmpdir):
    train_dataset = TFDSClassification(subset="train",
                                       tfds_name="cifar10",
                                       tfds_data_dir=str(tmpdir),
                                       tfds_download=True)

    validation_dataset = TFDSClassification(subset="validation",
                                            tfds_name="cifar10",
                                            tfds_data_dir=str(tmpdir),
                                            tfds_download=True)

    assert train_dataset.classes == ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    assert train_dataset.num_classes == 10
    assert train_dataset.num_per_epoch == 50000
    assert train_dataset.tf_dataset.output_shapes['image'].as_list() == [32, 32, 3]
    assert train_dataset.tf_dataset.output_shapes['label'].as_list() == [10]

    assert validation_dataset.classes == ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    assert validation_dataset.num_classes == 10
    assert validation_dataset.num_per_epoch == 10000
    assert validation_dataset.tf_dataset.output_shapes['image'].as_list() == [32, 32, 3]
    assert validation_dataset.tf_dataset.output_shapes['label'].as_list() == [10]
