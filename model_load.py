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
# from keras.backend import tf as ktf
from config import cfg
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.engine.topology import Network

from model import *


def model_create(dataset):
    #Function to create model and return
    emb_size, hidden_size  = cfg.TEXT.EMBEDDING_DIM, cfg.TEXT.HIDDEN_DIM // 2
    emb_size_dec, hidden_size_dec = cfg.TEXT.EMBEDDING_DIM_DEC, cfg.TEXT.HIDDEN_DIM_DEC
    vocab_size = dataset.n_words

    #DECODER I2T
    CR_model = CNN_ENCODER_RNN_DECODER(emb_size_dec, hidden_size_dec, vocab_size)
    if not cfg.TRAIN.RNN_DEC == '':
        CR_model.load_weights(cfg.TRAIN.RNN_DEC)
    CR_model.trainable = False
    # T2I
    RNN_model, words_emb, sent_emb, c_code = \
        RNN_ENCODER(emb_size, hidden_size,vocab_size, rec_unit=cfg.RNN_TYPE)
    netGs_out, out_image_block, attm, atts, z_code_input, mask_input = \
        G_DCGAN(sent_emb, words_emb, c_code)

    cap_input, eps_input = RNN_model.get_input_at(0)

    if cfg.TREE.BRANCH_NUM == 1:
        D_h_logits, D_hc_logits, D_pic_input = D_NET64(sent_emb)
    if cfg.TREE.BRANCH_NUM == 2:
        D_h_logits, D_hc_logits, D_pic_input = D_NET128(sent_emb)
    if cfg.TREE.BRANCH_NUM == 3:
        D_h_logits, D_hc_logits, D_pic_input = D_NET256(sent_emb)
    #For learning D
    D_model = Model([D_pic_input, cap_input],
                    [D_h_logits, D_hc_logits],
                    name="Discriminator")
    #D weight load
    if not cfg.TRAIN.NET_D == "":
        print("\nload D_model from" + cfg.TRAIN.NET_D)
        D_model.load_weights(cfg.TRAIN.NET_D)

    losses = {'D_h_out_layre': 'binary_crossentropy',
              'D_hc_out_layre': 'binary_crossentropy'}
    D_model.compile(
        loss = losses,
        loss_weights=[0.5, 0.5],
        optimizer=Adam(lr=cfg.TRAIN.D_LR, beta_1=cfg.TRAIN.D_BETA1),
        metrics=['accuracy'])

    #G (for D learning output)
    if cfg.TREE.BRANCH_NUM > 0:
        init_G_model = Model([cap_input, eps_input, z_code_input],
                             netGs_out[0],
                             name="init_G")
        G_output = init_G_model.output
        if not cfg.TRAIN.INIT_NET_G == "":
            init_G_model.load_weights(cfg.TRAIN.INIT_NET_G, by_name=True)

    if cfg.TREE.BRANCH_NUM > 1:
        next_G_model128 = Model(
            [cap_input, eps_input, z_code_input, mask_input],
            netGs_out[1],
            name="next_G128")
        G_output = next_G_model128.output
        if not cfg.TRAIN.NEXT128_NET_G == "":
            next_G_model128.load_weights(cfg.TRAIN.NEXT128_NET_G, by_name=True)

    if cfg.TREE.BRANCH_NUM > 2:
        next_G_model256 = Model(
            [cap_input, eps_input, z_code_input, mask_input],
            netGs_out[2],
            name="next_G256")
        G_output = next_G_model256.output
        if not cfg.TRAIN.NEXT256_NET_G == "":
            next_G_model256.load_weights(cfg.TRAIN.NEXT256_NET_G, by_name=True)
    #Coupling with output layer and weight load
    out_img = out_image_block(G_output)
    if cfg.TREE.BRANCH_NUM == 1:
        G_model = Model(init_G_model.get_input_at(0), out_img, name="Generator")
        if not cfg.TRAIN.INIT_NET_G == "":
            print("\nload G_model from" + cfg.TRAIN.INIT_NET_G)
            G_model.load_weights(cfg.TRAIN.INIT_NET_G)
    else:  #2 or 3
        G_model = Model(
            init_G_model.get_input_at(0) + [mask_input], out_img, name="Generator")
        if (not cfg.TRAIN.NEXT128_NET_G == "") and (cfg.TREE.BRANCH_NUM == 2):
            G_model.load_weights(cfg.TRAIN.NEXT128_NET_G)
        if (not cfg.TRAIN.NEXT256_NET_G == "") and (cfg.TREE.BRANCH_NUM == 3):
            G_model.load_weights(cfg.TRAIN.NEXT256_NET_G)

    #GRD (For learning G)
    DG_h_logits, DG_hc_logits = D_model([out_img, sent_emb])
    cr_cap_input = CR_model.get_input_at(0)[1]
    cr_logith = CR_model([out_img, cr_cap_input])  #pic_input #cap_input
    GRD_model = Model(
        G_model.get_input_at(0) + [cr_cap_input],
        [DG_h_logits, DG_hc_logits, cr_logith],
        name="GRD_model")


    # SIM for computing cosine similarity between real captions and generated captions
    image_input = CR_model.get_input_at(0)[0]
    caption_input = CR_model.get_input_at(0)[1]
    logit = CR_model([image_input, caption_input])
    cap_label_input = Input(shape=(cfg.TEXT.WORDS_NUM+1,vocab_size), name="cap_label_input")
    re_cap_label_input = Reshape((1, (cfg.TEXT.WORDS_NUM+1)*vocab_size))(cap_label_input)
    re_cr_logith = Reshape((1, (cfg.TEXT.WORDS_NUM+1)*vocab_size))(logit)
    cosin = Dot(axes=2, normalize=True)([re_cap_label_input, re_cr_logith])
    SIM_model = Model(CR_model.get_input_at(0) + [cap_label_input], [cosin], name="SIM_model")



    for lay in D_model.layers[:5]:
        lay.trainable = True
    for lay in D_model.layers[5:]:
        lay.trainable = False
        # lay.trainable = True
    GRD_model.compile(
        loss={
            'Discriminator': binary_crossentropy,
            'CNN_RNN_DEC': categorical_crossentropy
        },
        loss_weights=[0.5, 0.5, cfg.TRAIN.RNN_DEC_LOSS_W],
        optimizer=Adam(lr=cfg.TRAIN.G_LR, beta_1=cfg.TRAIN.G_BETA1),
        metrics=['accuracy'])

    return G_model, D_model, GRD_model, CR_model, RNN_model, SIM_model


def model_create_pretrain(dataset):
    #Function to create model and return
    #DECODER for learning

    emb_size, hidden_size = cfg.TEXT.EMBEDDING_DIM, cfg.TEXT.HIDDEN_DIM // 2
    emb_size_dec, hidden_size_dec = cfg.TEXT.EMBEDDING_DIM_DEC, cfg.TEXT.HIDDEN_DIM_DEC
    vocab_size = dataset.n_words

    #DECODER
    CR_model = CNN_ENCODER_RNN_DECODER(emb_size_dec, hidden_size_dec,
                                       vocab_size)
    if not cfg.TRAIN.RNN_DEC == '':
        CR_model.load_weights(cfg.TRAIN.RNN_DEC)
    CR_model.trainable = True

    CR_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=cfg.TRAIN.DEC_LR),
        metrics=['accuracy'])

    return CR_model

def model_create_new(dataset):
    #Function to create model and return
    emb_size, hidden_size  = cfg.TEXT.EMBEDDING_DIM, cfg.TEXT.HIDDEN_DIM // 2
    emb_size_dec, hidden_size_dec = cfg.TEXT.EMBEDDING_DIM_DEC, cfg.TEXT.HIDDEN_DIM_DEC
    vocab_size = dataset.n_words

    #DECODER I2T
    CR_model = CNN_ENCODER_RNN_DECODER(emb_size_dec, hidden_size_dec, vocab_size)
    if not cfg.TRAIN.RNN_DEC == '':
        CR_model.load_weights(cfg.TRAIN.RNN_DEC)
    CR_model.trainable = False
    # T2I
    RNN_model, words_emb, sent_emb, c_code = \
        RNN_ENCODER(emb_size, hidden_size,vocab_size, rec_unit=cfg.RNN_TYPE)
    netGs_out, out_image_block, attm, atts, z_code_input, mask_input = \
        G_DCGAN(sent_emb, words_emb, c_code)

    cap_input, eps_input = RNN_model.get_input_at(0)

    if cfg.TREE.BRANCH_NUM == 1:
        D_h_logits, D_hc_logits, D_pic_input = D_NET64(sent_emb)
    if cfg.TREE.BRANCH_NUM == 2:
        D_h_logits, D_hc_logits, D_pic_input = D_NET128(sent_emb)
    if cfg.TREE.BRANCH_NUM == 3:
        D_h_logits, D_hc_logits, D_pic_input = D_NET256(sent_emb)
    #For learning D
    D_model = Model([D_pic_input, cap_input],
                    [D_h_logits, D_hc_logits],
                    name="Discriminator")
    #D weight load
    if not cfg.TRAIN.NET_D == "":
        print("\nload D_model from" + cfg.TRAIN.NET_D)
        D_model.load_weights(cfg.TRAIN.NET_D)
    D_model.compile(
        loss={'D_h_out_layre': 'binary_crossentropy',
              'D_hc_out_layre': 'binary_crossentropy'},
        loss_weights=[0.2, 0.8],
        optimizer=Adam(lr=cfg.TRAIN.D_LR, beta_1=cfg.TRAIN.D_BETA1),
        metrics=['accuracy'])

    n_D_trainable  = len(D_model.trainable_weights)

    D_model_freeze = Network([D_pic_input, cap_input],
                    [D_h_logits, D_hc_logits],
                    name="Discriminator")


    #G (for D learning output)
    if cfg.TREE.BRANCH_NUM > 0:
        init_G_model = Model([cap_input, eps_input, z_code_input],
                             netGs_out[0],
                             name="init_G")
        G_output = init_G_model.output
        if not cfg.TRAIN.INIT_NET_G == "":
            init_G_model.load_weights(cfg.TRAIN.INIT_NET_G, by_name=True)

    if cfg.TREE.BRANCH_NUM > 1:
        next_G_model128 = Model(
            [cap_input, eps_input, z_code_input, mask_input],
            netGs_out[1],
            name="next_G128")
        G_output = next_G_model128.output
        if not cfg.TRAIN.NEXT128_NET_G == "":
            next_G_model128.load_weights(cfg.TRAIN.NEXT128_NET_G, by_name=True)

    if cfg.TREE.BRANCH_NUM > 2:
        next_G_model256 = Model(
            [cap_input, eps_input, z_code_input, mask_input],
            netGs_out[2],
            name="next_G256")
        G_output = next_G_model256.output
        if not cfg.TRAIN.NEXT256_NET_G == "":
            next_G_model256.load_weights(cfg.TRAIN.NEXT256_NET_G, by_name=True)
    #Coupling with output layer and weight load
    out_img = out_image_block(G_output)
    if cfg.TREE.BRANCH_NUM == 1:
        G_model = Model(init_G_model.get_input_at(0), out_img, name="Generator")
        if not cfg.TRAIN.INIT_NET_G == "":
            print("\nload G_model from" + cfg.TRAIN.INIT_NET_G)
            G_model.load_weights(cfg.TRAIN.INIT_NET_G)
    else:  #2 or 3
        G_model = Model(
            init_G_model.get_input_at(0) + [mask_input], out_img, name="Generator")
        if (not cfg.TRAIN.NEXT128_NET_G == "") and (cfg.TREE.BRANCH_NUM == 2):
            G_model.load_weights(cfg.TRAIN.NEXT128_NET_G)
        if (not cfg.TRAIN.NEXT256_NET_G == "") and (cfg.TREE.BRANCH_NUM == 3):
            G_model.load_weights(cfg.TRAIN.NEXT256_NET_G)

    n_G_trainable = len(G_model.trainable_weights)

    #GRD (For learning G)
    D_model_freeze.trainable = False
    DG_h_logits, DG_hc_logits = D_model_freeze([out_img, sent_emb])
    cr_cap_input = CR_model.get_input_at(0)[1]
    cr_logith = CR_model([out_img, cr_cap_input])  #pic_input #cap_input
    GRD_model = Model(
        G_model.get_input_at(0) + [cr_cap_input],
        [DG_h_logits, DG_hc_logits, cr_logith],
        name="GRD_model")
    GRD_model.compile(
        loss={
            'Discriminator': binary_crossentropy,
            'CNN_RNN_DEC': categorical_crossentropy
        },
        loss_weights=[0.5, 0.5, cfg.TRAIN.RNN_DEC_LOSS_W],
        optimizer=Adam(lr=cfg.TRAIN.G_LR, beta_1=cfg.TRAIN.G_BETA1),
        metrics=['accuracy'])

    if not len(D_model._collected_trainable_weights) == n_D_trainable:
        print("\n D collected trainable weights {:,} does not equals trainable weights {:,}!".format(len(D_model._collected_trainable_weights), n_D_trainable))
    if not len(GRD_model._collected_trainable_weights) == n_G_trainable:
        print("\n G collected trainable weights {:,} does not equals trainable weights {:,}!".format(len(GRD_model._collected_trainable_weights), n_G_trainable))

    # SIM for computing cosine similarity between real captions and generated captions
    image_input = CR_model.get_input_at(0)[0]
    caption_input = CR_model.get_input_at(0)[1]
    logit = CR_model([image_input, caption_input])
    cap_label_input = Input(shape=(cfg.TEXT.WORDS_NUM + 1, vocab_size), name="cap_label_input")
    re_cap_label_input = Reshape((1, (cfg.TEXT.WORDS_NUM + 1) * vocab_size))(cap_label_input)
    re_cr_logith = Reshape((1, (cfg.TEXT.WORDS_NUM + 1) * vocab_size))(logit)
    cosin = Dot(axes=2, normalize=True)([re_cap_label_input, re_cr_logith])
    SIM_model = Model(CR_model.get_input_at(0) + [cap_label_input], [cosin], name="SIM_model")


    return G_model, D_model, GRD_model, CR_model, RNN_model, SIM_model