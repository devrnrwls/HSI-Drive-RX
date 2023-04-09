import argparse
import os
import numpy as np
import sys
import gc
import json
import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Dataset
import Generator
from metrics import CustomSparseCategoricalAccuracy, CustomSparseCategoricalPrecision, CustomSparseCategoricalIoU
import unet as unet

gc.collect()

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]


def test_model(net_config, dataset_config, pred_dir, float_model, exp_number, applyThresholdToPredictions, threshold, implicit_norm, usePatches=False, data_augmentation=False):

    K.set_learning_phase(0)
    normMaxValue = dataset_config["maximo_global"]
    normMinValue = dataset_config["minimo_global"]
    training_model_options = net_config['training_model_options']
    test_model_options = net_config['test_model_options']

    eval_dir = dataset_config['test_dir']
    weights = np.array(dataset_config['weights' + str(exp_number)])
    eval_dataset = Dataset.HSIDriveDataset(eval_dir, dataset_config['data_folder_test'], dataset_config['label_folder_test' + str(exp_number)], weights, normMaxValue, normMinValue, usePatches=usePatches)
    test_batch_size = test_model_options["batch_size"]
    dataset_generator = Generator.DataGenerator(eval_dataset, dim=(training_model_options["patch_height"], training_model_options["patch_width"]), batch_size=test_model_options["batch_size"], shuffle=False, n_channels = 25)

    num_batches = len(eval_dataset) / test_batch_size
    if ((len(eval_dataset) % test_batch_size) != 0):
        raise("Test batch size should be modified to be a divisor of the length of the test dataset")

    # Load trained floating-point model (ojo, hago Inference=False si quiero umbrales)
    trained_model = unet.UNET(training_model_options['patch_height'], training_model_options['patch_width'], training_model_options['input_channels'], dataset_config['num_clases_exp' + str(exp_number)] + 1, "channels_last", batchnorm=True, enc_conv_kernel_size=net_config["training_model_options"]["enc_conv_filter_size"], enc_depth=net_config["training_model_options"]["encoder_depth"], n_filters=net_config["training_model_options"]["initial_number_of_filters"], dropout=0.5, implicitNorm=implicit_norm, inference=True, dataAugmentation=data_augmentation)

    #Esto tendría que saber hacerlo de una manera más elegante y cargar el modelo directamente. Creo que es porque lo guardo como h5 en vez de SavedModel.
    #No afecta la elección de estos dos valores
    loss_fcn = SparseCategoricalCrossentropy()
    optimizer_fcn = Adam()
    #Estos sí obviamenete
    trained_model.load_weights(float_model)
    trained_model.compile(optimizer_fcn, loss=loss_fcn, metrics=[CustomSparseCategoricalAccuracy(), CustomSparseCategoricalPrecision(), CustomSparseCategoricalIoU()], weighted_metrics=[CustomSparseCategoricalAccuracy(), CustomSparseCategoricalPrecision(), CustomSparseCategoricalIoU()])
    trained_model.evaluate(dataset_generator)

    for j in range(int(num_batches)):
        imgs, _, _ = dataset_generator[j] #Si el tamaño del batch es el dataset completo solo se ejecuta una vez
        segs = trained_model.predict(imgs)
        preds = np.argmax(segs, axis=-1)

        if applyThresholdToPredictions:
            maxValue = np.amax(segs, axis=-1)
            print(maxValue)
            preds[maxValue<threshold] = 0

        src_path = os.path.dirname(os.path.abspath(__file__))
        for i, pred in enumerate(preds):
            _, label_name = eval_dataset.getitemname(i + test_batch_size*j)
            aux = findOccurrences(label_name, '/') #Quiero extraer cierta parte del nombre completo (el más 1 es porque me interesa la posición siguiente)
            patch_name = label_name[aux[len(aux) - 1] + 1:len(label_name)]
            patch_path = os.path.join(src_path, pred_dir, patch_name)
            cv2.imwrite(patch_path, pred)

    return


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-fm', '--float_model',                     type=str, default='./model_name.h5')
    ap.add_argument('-pd', '--pred_dir',                        type=str, default='../dataset/Pred/')
    ap.add_argument('-ex', '--exp_number',                      type=int, default=1)
    ap.add_argument('-at', '--applyThresholdToPredictions',     action='store_true')
    ap.add_argument('-t',  '--threshold',                       type=float, default=0.8)
    ap.add_argument('-i',  '--implicit_norm',                   action='store_true')
    args = ap.parse_args()

    try:
        src_path = os.path.dirname(os.path.abspath(__file__))
        net_config_path = os.path.join(src_path, '../UNet.json')
        with open(net_config_path, 'r') as f:
            net_config = json.load(f)
    except:
        sys.exit('The model name is incorrect!')

    try:
        src_path = os.path.dirname(os.path.abspath(__file__))
        dataset_config_path = os.path.join(src_path, '../Dataset.json')
        with open(dataset_config_path, 'r') as g:
            dataset_config = json.load(g)
        g.close()
        with open(dataset_config_path, 'w+') as f:
            dataset_config["versionDASIP"]["weights"] = dataset_config["versionDASIP"]["weights" + str(args.exp_number)]
            dataset_config["versionDASIP"]["global_weights"] = dataset_config["versionDASIP"]["global_weights" + str(args.exp_number)]
            f.write(json.dumps(dataset_config, indent=0))
        f.close()
    except:
        sys.exit('The dataset name is incorrect!')

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print('Command line options:')
    print(' --float_model                   : ', args.float_model)
    print(' --pred_dir                      : ', args.pred_dir)
    print(' --exp_number                    : ', args.exp_number)
    print(' --applyThresholdToPredictions   : ', args.applyThresholdToPredictions)
    print(' --threshold                     : ', args.threshold)
    print(' --implicit_norm                 : ', args.implicit_norm)
    print('------------------------------------\n')

    test_model(net_config, dataset_config["versionDASIP"], args.pred_dir, args.float_model, args.exp_number, args.applyThresholdToPredictions, args.threshold, args.implicit_norm)

if __name__ ==  "__main__":
    main()
