from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Input, Concatenate, Lambda
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, Dropout
from keras.layers import Reshape, LeakyReLU, ZeroPadding2D
from keras.layers import Conv1D, Add, Conv2D, UpSampling2D
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
import keras
from keras.optimizers import Adam
from keras.backend import tf as ktf
from config import cfg
import tensorflow as tf
from dataset import TextDataset
from dataset import TextDataset_Pascal
from generator import DataGenerator
from generator import DataGenerator_all
from generator import DataGenerator_class

from model import *
from model_load import model_create
from keras.losses import categorical_crossentropy, binary_crossentropy
import torchvision.transforms as transforms
from copy import deepcopy
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime


def main():
    num = 20
    #DataGenerator
    imsize = cfg.TREE.BASE_SIZE * (2**(cfg.TREE.BRANCH_NUM - 1))  #64, 3
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])
    cfg.DATA_DIR = "data/pascal"

    dataset_train = TextDataset_Pascal(
        cfg.DATA_DIR,
        "train",
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform)
    assert dataset_train

    traingenerator = DataGenerator(dataset_train, batchsize=cfg.TRAIN.BATCH_SIZE)

    ##Create model
    G_model, D_model, GRD_model, CR_model, RNN_model, SIM_model = model_create(dataset_train)
    print("loadmodel_completed")


    image_list, captions_ar, captions_ar_prezeropad, \
    z_code, eps_code, mask, keys_list, captions_label, \
    real_label, fake_label = traingenerator.sample(num)
    # for image plot
    test_noise = deepcopy(z_code)
    test_eps = deepcopy(eps_code)
    test_cap_pd = deepcopy(captions_ar_prezeropad)
    test_mask = deepcopy(mask)
    test_mask = np.where(test_mask == 1, -float("inf"), 0)
    test_real_image = deepcopy(image_list[0])

    sample_images(test_noise, test_eps, test_cap_pd, test_real_image, test_mask, G_model)




def sample_images(noise, eps, cap_pd, test_real_image, mask, G_model):
    r, c = 5, 4
    if cfg.TREE.BRANCH_NUM == 1:
        gen_imgs = G_model.predict([cap_pd, eps, noise])
    else:
        gen_imgs = G_model.predict([cap_pd, eps, noise, mask])
    # Rescale images
    gen_imgs = (gen_imgs * 127.5 + 127.5).astype("int")
    real_imgs = (test_real_image * 127.5 + 127.5).astype("int")
    fig, axs = plt.subplots(r, 2*c)
    cnt = 0
    for i in range(r):
        for j in range(c):

            axs[i, 2*j].imshow(real_imgs[cnt])
            axs[i, 2*j].axis('off')
            axs[i, 2*j+1].imshow(gen_imgs[cnt])
            axs[i, 2*j+1].axis('off')
            cnt += 1
    fig.savefig("generated_imgs/real_gen.png")
    plt.close()
if __name__ == '__main__':
    main()