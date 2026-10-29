import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
