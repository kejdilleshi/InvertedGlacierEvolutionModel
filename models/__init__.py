# Compatibility module for loading old checkpoints
# This redirects imports from the old 'models' module to the new 'core.cnn_model' module
from core.cnn_model import CNN

__all__ = ['CNN']
