"""Convenience function to create, save, and load models."""

import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.io import gfile

# pylint: disable=g-bad-import-order
from datadrivenpdes.core import grids
from datadrivenpdes.core import models
from datadrivenpdes.advection import equations as advection_equations
# pylint: enable=g-bad-import-order

from typing import Any, Mapping

def model_from_config(grid, config: Mapping[str, Any]):
  """Construct a ML model from a configuration dict."""
  model = models.PseudoLinearModel(
      advection_equations.FiniteVolumeAdvection(0.5),
      grid,
      scaled_keys={'concentration'},
      core_model_func=models.RescaledConv2DStack,
      **config,
      )

  return model


def save_weights(model, path, overwrite=True):
  """Customized version of Keras model.save_weights().

  - Works with both local and remote storage.
  - Creates intermediate directories if missing.
  """
  tmp_dir = tempfile.mkdtemp()
  dirname, basename = os.path.split(path)
  tmp_path = os.path.join(tmp_dir, basename)

  model.save_weights(tmp_path)
  gfile.makedirs(dirname)
  gfile.copy(tmp_path, path, overwrite=overwrite)
  gfile.remove(tmp_path)


def load_weights(model, path, initialize_weights=True):
  """Customized version of Keras model.load_weights().

  - Works with both local and remote storage.
  - Automatically initialize weights before loading
  """
  if initialize_weights:
    # cannot use Keras model.build(), because model input is a dict of tensors
    fake_data = np.zeros([1, model.grid.size_x, model.grid.size_y],
                         dtype=np.float32)
    init_state = {k: fake_data for k in model.equation.base_keys}
    model.call(init_state)

  tmp_dir = tempfile.mkdtemp()
  _, basename = os.path.split(path)
  tmp_path = os.path.join(tmp_dir, basename)

  gfile.copy(path, tmp_path)
  model.load_weights(tmp_path)
  gfile.remove(tmp_path)


class ModelCheckpoint(tf.keras.callbacks.Callback):
  """Customized version of keras.callbacks.ModelCheckpoint.

  - Works with both local and remote storage.
  - Creates intermediate directories if missing.
  """
  def __init__(self, filepath, period=1, verbose=True):
    """
    Args:
      filepath: str, path for model weights file.
        {epoch} and {loss} will be replaced by actual values.
        An example is './directory/weights_epoch{epoch:02d}_loss{loss:.4f}.h5'
      period: int, frequency to save model.
      verbose: bool, whether to print weight file name when saving.
    """
    self.filepath = filepath
    self.period = period
    self.verbose = verbose
    super().__init__()

  def on_epoch_end(self, epoch, logs):

    if (epoch+1) % self.period == 0:
      realpath = self.filepath.format(epoch=epoch+1, loss=logs['loss'])
      if self.verbose:
        print('\nSave model weights to:', realpath)
      save_weights(self.model, realpath)
