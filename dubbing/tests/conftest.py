"""pytest configuration for dubbing/tests.

GPU selection
-------------
All GPU-using tests share a single device, controlled by the environment
variable ``TEST_GPU`` (default: ``"0"``).  Set it before running pytest to
redirect to a different card::

    TEST_GPU=2 pytest dubbing/tests

The variable is applied here at collection time so every test module
inherits it regardless of import order.
"""

import os

_TEST_GPU = os.environ.get("TEST_GPU", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = _TEST_GPU
