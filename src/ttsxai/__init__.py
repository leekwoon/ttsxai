import os

d = os.path.dirname

PACKAGE_DIR = d(d(d(__file__)))
PRETRAINED_MODELS_DIR = os.path.join(PACKAGE_DIR, 'pretrained_models')
