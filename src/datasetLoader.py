import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

ds = tfds.load('mnist', split='train', shuffle_files=True)