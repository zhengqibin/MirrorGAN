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
import tensorflow as tf
from dataset import TextDataset
from dataset import TextDataset_Pascal
from generator import DataGenerator
from generator import DataGenerator_all
from generator import DataGenerator_class

from model import *
from model_load import model_create_new
from keras.losses import categorical_crossentropy, binary_crossentropy
import torchvision.transforms as transforms
from copy import deepcopy
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime


def main():
    #DataGenerator
    trainable = cfg.Train
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

    dataset_test = TextDataset_Pascal(
        cfg.DATA_DIR,
        "test",
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform)
    assert dataset_test

    traingenerator = DataGenerator(dataset_train, batchsize=cfg.TRAIN.BATCH_SIZE)

    ##Create model
    G_model, D_model, GRD_model, CR_model, RNN_model, SIM_model = model_create_new(dataset_train)
    print("loadmodel_completed")



    if trainable:
        # Record Tensorboard Log
        sess = tf.Session()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)
        D_LOSS = tf.placeholder(tf.float32, [])
        D_LOSS_wrong = tf.placeholder(tf.float32, [])
        D_ACC_realism = tf.placeholder(tf.float32, [])
        D_ACC_consistency = tf.placeholder(tf.float32, [])
        D_ACC_wong_consistency = tf.placeholder(tf.float32, [])
        D_ACC_wong_realism = tf.placeholder(tf.float32, [])

        G_LOSS = tf.placeholder(tf.float32, [])
        G_LOSS_enc = tf.placeholder(tf.float32, [])


        tf.summary.scalar("D_LOSS", D_LOSS)
        tf.summary.scalar("D_LOSS_wrong", D_LOSS_wrong)
        tf.summary.scalar("D_ACC_realism", D_ACC_realism)
        tf.summary.scalar("D_ACC_consitency", D_ACC_consistency)
        tf.summary.scalar("D_ACC_wong_consistency", D_ACC_wong_consistency)
        tf.summary.scalar("D_ACC_wong_realism", D_ACC_wong_realism)
        tf.summary.scalar("G_LOSS", G_LOSS)
        tf.summary.scalar("G_LOSS_enc", G_LOSS_enc)
        merged = tf.summary.merge_all()
        # Preparation for learning
        total_epoch = cfg.TRAIN.MAX_EPOCH
        batch_size = traingenerator.batchsize
        step_epoch = int(len(dataset_train) / batch_size)
        wrong_step = 1
        wrong_step_epoch = int(step_epoch / wrong_step)
        eval_step = 20

        image_list, captions_ar, captions_ar_prezeropad, \
        z_code, eps_code, mask, keys_list, captions_label, \
        real_label, fake_label = next(traingenerator)
        traingenerator.count = 0
        # for image plot
        test_noise = deepcopy(z_code[:20])
        test_eps = deepcopy(eps_code[:20])
        test_cap_pd = deepcopy(captions_ar_prezeropad[:20])
        test_cap = deepcopy(captions_ar[:20])
        test_mask = deepcopy(mask[:20])
        test_mask = np.where(test_mask == 1, -float("inf"), 0)
        test_real_image = deepcopy(image_list[0][:20])

        # Start learning
        print("batch_size: {}  step_epoch : {} wrong_step_epoch {}".format(
            batch_size, step_epoch, wrong_step_epoch))

        for epoch in range(total_epoch):
            total_D_loss = 0
            total_D_acc = 0
            total_D_acc_realism = 0
            total_D_acc_consitency = 0
            total_D_wrong_loss = 0
            total_D_wrong_acc = 0
            total_D_wrong_acc_realism = 0
            total_D_wrong_acc_consitency = 0
            total_G_loss = 0
            total_G_des_loss = 0
            total_G_enc_loss = 0

            print("----------------EPOCH: {} START----------------".format(epoch))

            for batch in tqdm(range(step_epoch)):
                image_list, captions_ar, captions_ar_prezeropad, \
                    z_code, eps_code, mask, keys_list, captions_label, \
                        real_label, fake_label = next(traingenerator)

                mask = np.where(mask == 1, -float("inf"), 0)

                if cfg.TREE.BRANCH_NUM == 1:
                    real_image = image_list[0]
                if cfg.TREE.BRANCH_NUM == 2:
                    real_image = image_list[1]
                if cfg.TREE.BRANCH_NUM == 3:
                    real_image = image_list[2]
                num_image = len(real_image)
                #D learning

                if cfg.TREE.BRANCH_NUM == 1:
                    fake_image = G_model.predict(
                        [captions_ar_prezeropad, eps_code, z_code])
                else:  # 2 or 3
                    fake_image = G_model.predict(
                        [captions_ar_prezeropad, eps_code, z_code, mask])

                if batch % 1 == 0:
                    histDr = D_model.train_on_batch(
                        [real_image, captions_ar_prezeropad],
                        [real_label, real_label],
                    )
                    # histDr = D_model.test_on_batch(
                    #     [real_image, captions_ar_prezeropad],
                    #     [real_label, real_label],
                    # )

                    total_D_loss += histDr[0]
                    total_D_acc_realism += histDr[3]
                    total_D_acc_consitency += histDr[4]

                    total_D_acc += (histDr[3] + histDr[4]) / 2

                    histDf = D_model.train_on_batch(
                        [fake_image, captions_ar_prezeropad],
                        [fake_label, fake_label],
                    )
                    # histDf = D_model.test_on_batch(
                    #     [fake_image, captions_ar_prezeropad],
                    #     [fake_label, fake_label],
                    # )
                    total_D_loss += histDf[0]
                    total_D_acc += (histDf[3] + histDf[4]) / 2

                if batch % wrong_step == 0:
                    histDw = D_model.train_on_batch(
                        [real_image[:-1], captions_ar_prezeropad[1:]],
                        [real_label[:-1], fake_label[:-1]],
                    )
                    temp2 = D_model.predict([real_image[:-1], captions_ar_prezeropad[1:]])
                    # histDw = D_model.test_on_batch(
                    #     [real_image[:-1], captions_ar_prezeropad[1:]],
                    #     [real_label[:-1], fake_label[:-1]],
                    # )

                    total_D_wrong_loss += histDw[0]
                    total_D_wrong_acc_realism += histDw[3]
                    total_D_wrong_acc_consitency += (histDw[4])

                #G learning

                if cfg.TREE.BRANCH_NUM == 1:
                    histGRD = GRD_model.train_on_batch(
                        [captions_ar_prezeropad, eps_code, z_code, captions_ar],
                        [real_label, real_label, captions_label],
                    )
                    # histGRD = GRD_model.test_on_batch(
                    #     [captions_ar_prezeropad, eps_code, z_code, captions_ar],
                    #     [real_label, real_label, captions_label],
                    # )
                else:  # 2 or 3
                    histGRD = GRD_model.train_on_batch(
                        [captions_ar_prezeropad, eps_code, z_code, mask, captions_ar],
                        [real_label, real_label, captions_label],g))
                    )
                    # histGRD = GRD_model.test_on_batch(
                    #     [captions_ar_prezeropad, eps_code, z_code, mask, captions_ar],
                    #     [real_label, real_label, captions_label],
                    # )
                total_G_loss += histGRD[0]
                total_G_des_loss += (histGRD[1] + histGRD[2]) / 2
                total_G_enc_loss += histGRD[3]
            #Calculation of loss
            D_loss = total_D_loss / step_epoch / 2
            D_acc = total_D_acc / step_epoch / 2
            D_wrong_loss = total_D_wrong_loss / wrong_step_epoch
            D_wrong_acc = total_D_wrong_acc / wrong_step_epoch
            G_loss = total_G_loss / step_epoch/ (cfg.TRAIN.RNN_DEC_LOSS_W + 1)
            G_des_loss = total_G_des_loss / step_epoch
            G_enc_loss = total_G_enc_loss / step_epoch / cfg.TRAIN.RNN_DEC_LOSS_W

            D_wrong_acc_realism = total_D_wrong_acc_realism / step_epoch
            D_wrong_acc_consistency = total_D_wrong_acc_consitency / wrong_step_epoch
            D_acc_realism = total_D_acc_realism / step_epoch
            D_acc_consistency = total_D_acc_consitency / step_epoch

            if epoch % 1 == 0:
                summary = sess.run(merged, feed_dict={D_LOSS: D_loss,
                                                      D_LOSS_wrong: D_wrong_loss,
                                                      D_ACC_realism: D_acc_realism,
                                                      D_ACC_consistency: D_acc_consistency,
                                                      D_ACC_wong_consistency: D_wrong_acc_realism,
                                                      D_ACC_wong_realism: D_wrong_acc_consistency,
                                                      G_LOSS: G_loss,
                                                      G_LOSS_enc: G_enc_loss
                                                      })
                writer.add_summary(summary, epoch)

            print(
                "\nD_loss: {:.5f} D_wrong_loss: {:.5f} D_wrong_acc_realism:  {:.5f} D_wrong_acc_consistency:  {:.5f} D_acc_realism:  {:.5f} D_acc_consistency:  {:.5f} "
                .format(D_loss, D_wrong_loss, D_wrong_acc_realism, D_wrong_acc_consistency, D_acc_realism, D_acc_consistency))
            print(
                "G_loss:  {:.5f} G_discriminator_loss:  {:.5f} G_encoder_loss:  {:.5f}"
                .format(G_loss, G_des_loss, G_enc_loss))



            # Save model
            if (epoch % 10 == 0) or (epoch == total_epoch-1):
                G_save_path = "model/G_epoch{}.h5".format(epoch)
                G_model.save_weights(G_save_path)
                D_save_path = "model/D_epoch{}.h5".format(epoch)
                D_model.save_weights(D_save_path)
            #Save image
            if epoch % 1 == 0:
                sample_images(epoch, test_noise, test_eps, test_cap_pd, test_real_image, test_mask, G_model)
    # Calculate conditional probability
    if not trainable:
        # conditional_prob_all(dataset_pascal_test, D_model, GRD_model, SIM_model)
        conditional_prob_class(dataset_train, cfg.DATASET_NAME, D_model, GRD_model, SIM_model)



def conditional_prob_class(dataset, dataset_name, D_model, GRD_model, SIM_Model):
    data_generator = DataGenerator_class(dataset, batchsize=cfg.TRAIN.BATCH_SIZE)

    for class_idx in tqdm(range(20)):
        image_list, captions_ar, captions_ar_prezeropad, \
        z_code, eps_code, mask, keys_list, captions_label, \
        real_label, fake_label, cls_label = next(data_generator)
        length = captions_ar.shape[0]
        if length == 0:
            continue
        if cfg.TREE.BRANCH_NUM == 1:
            real_image = image_list[0]
        if cfg.TREE.BRANCH_NUM == 2:
            real_image = image_list[1]
        if cfg.TREE.BRANCH_NUM == 3:
            real_image = image_list[2]
        Prob_IT = np.zeros((length, length))
        Prob_TI = np.zeros((length, length))
        for index1, img in enumerate(real_image):
            img_array = np.tile(img, (length,1,1,1))
            res_TI = D_model.predict([img_array, captions_ar_prezeropad])
            num_TI = res_TI[1].shape[0]
            Prob_TI[index1, :] = res_TI[1].reshape(num_TI)
            res_IT = SIM_Model.predict([img_array, captions_ar_prezeropad, captions_label])
            num_IT = res_IT.shape[0]
            Prob_IT[index1, :] = res_IT.reshape([num_IT])

        np.savetxt('result/'+dataset_name+'/class/IT_class'+ str(cls_label) +'.csv', Prob_IT, delimiter=',')
        np.savetxt('result/'+dataset_name+'/class/TI_class'+ str(cls_label) +'.csv', Prob_TI, delimiter=',')


def conditional_prob_all(dataset, D_model, GRD_model, SIM_Model):
    data_generator = DataGenerator_all(dataset, batchsize=cfg.TRAIN.BATCH_SIZE)

    image_list, captions_ar, captions_ar_prezeropad, \
    z_code, eps_code, mask, keys_list, captions_label, \
    real_label, fake_label, cls_label = next(data_generator)
    mask = np.where(mask == 1, -float("inf"), 0)
    length = captions_ar.shape[0]
    if cfg.TREE.BRANCH_NUM == 1:
        real_image = image_list[0]
    if cfg.TREE.BRANCH_NUM == 2:
        real_image = image_list[1]
    if cfg.TREE.BRANCH_NUM == 3:
        real_image = image_list[2]
    Prob_IT = np.zeros((length, length))
    Prob_TI = np.zeros((length, length))
    for index1, img in enumerate(tqdm(real_image)):
        img_array = np.tile(img, (length, 1, 1, 1))
        res_TI = D_model.predict([img_array, captions_ar_prezeropad])
        num_TI = res_TI[1].shape[0]
        Prob_TI[index1, :] = res_TI[1].reshape(num_TI)
        res_IT = SIM_Model.predict([img_array, captions_ar_prezeropad, captions_label])
        num_IT = res_IT.shape[0]
        Prob_IT[index1, :] = res_IT.reshape([num_IT])

    np.savetxt('result/sim_IT_all.csv', Prob_IT, delimiter=',')
    np.savetxt('result/sim_TI_all.csv', Prob_TI, delimiter=',')

    r1, r5, r10, r50 = recall_at_k(Prob_IT)
    print("Recall @1, 5, 10, 50: ")
    print("Image to Text: %.1f %.1f %.1f %.1f " % (r1, r5, r10, r50))
    print("-----------------------------------")

    r1, r5, r10, r50 = recall_at_k(Prob_TI)
    print("Text to Image: %.1f %.1f %.1f %.1f " % (r1, r5, r10, r50))
    print("-----------------------------------")





def recall_at_k(sim_matrix):
    indexs = np.argsort(-sim_matrix, axis=1)
    ranks = np.zeros([len(indexs), 1])
    for i in range(len(indexs)):
        index = indexs[i, :]
        for j in range(len(index)):
            if i == index[j]:
                ranks[i] = j
                break

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)


    return r1, r5, r10, r50

def sample_images(epoch, noise, eps, cap_pd, test_real_image, mask, G_model):
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
    fig.savefig("gan_img/real_gen_%d.png" % epoch)
    plt.close()
if __name__ == '__main__':
    main()
