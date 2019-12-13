from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C
__C.Train = True
# Dataset name: flowers, birds
__C.DATASET_NAME = 'pascal'
__C.CONFIG_NAME = ''
__C.DATA_DIR = 'data/' + __C.DATASET_NAME

__C.RNN_TYPE = 'gru'  # 'lstm'

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 1
__C.TREE.BASE_SIZE = 64

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 20
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.FLAG = True
#load model path（weight onry）
if __C.Train == True:
    __C.TRAIN.NET_D = ''
    __C.TRAIN.INIT_NET_G = ''
    # __C.TRAIN.NET_D = 'model/pascal/pre-train/D_epoch605.h5'
    # __C.TRAIN.INIT_NET_G = 'model/pascal/pre-train/G_epoch605.h5'
else:
    __C.TRAIN.NET_D = 'model/sim/D_epoch599.h5'
    __C.TRAIN.INIT_NET_G = 'model/sim/G_epoch599.h5'
    # __C.TRAIN.NET_D = ''
    # __C.TRAIN.INIT_NET_G = ''

__C.TRAIN.NEXT128_NET_G = ''
__C.TRAIN.NEXT256_NET_G = ''
__C.TRAIN.RNN_DEC = 'model/' + __C.DATASET_NAME+ '/cnn_rnn_encoder.h5'
# __C.TRAIN.RNN_DEC = ''

# __C.TRAIN.D_LR = 0.0005  #D learning rate
__C.TRAIN.D_LR = 0.001  #D learning rate
__C.TRAIN.D_BETA1 = 0.5  #ADAM, beta1
# __C.TRAIN.G_LR = 0.0004  #G learning rate
__C.TRAIN.G_LR = 0.001  #G learning rate
__C.TRAIN.G_BETA1 = 0.5  #ADAM, beta1
__C.TRAIN.RNN_DEC_LOSS_W = 0.1  #decoder_RNN loss_weight
# __C.TRAIN.RNN_DEC_LOSS_W = 1  #decoder_RNN loss_weight

__C.TRAIN.DEC_LR = 0.001  #pretrain decoder_RNN earning rate
# __C.TRAIN.DEC_LR = 0.005  #pretrain decoder_RNN earning rate
__C.TRAIN.DEC_SAVE_PATH = 'model/'+__C.DATASET_NAME + '/cnn_rnn_encoder.h5'
__C.TRAIN.DEC_MAX_EPOCH = 20

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
# dimension of the random noise
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False


__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
if __C.DATASET_NAME == 'pascal':
    __C.TEXT.CAPTIONS_PER_IMAGE = 1
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.EMBEDDING_DIM_DEC = 512
__C.TEXT.HIDDEN_DIM = 256
__C.TEXT.HIDDEN_DIM_DEC = 512
__C.TEXT.WORDS_NUM = 18


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():  #変更python3
        # a must specify keys that are in b
        if not k in b: #変更python3
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)