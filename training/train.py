import argparse
import os
import numpy as np
import sys
import json
import gc
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unet as unet
import Dataset
import Generator
from metrics import CustomSparseCategoricalAccuracy, CustomSparseCategoricalPrecision, CustomSparseCategoricalIoU


# datetime object containing current date and time
init = datetime.now()
init_string = init.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", init_string)

gc.collect()

def lr_time_based_decay(epoch, lr):
    if epoch == 30:
        actual_lr = lr * 0.1
    else:
        actual_lr = lr
    return actual_lr

def train_model(net_config, dataset_config, float_model, exp_number, implicit_norm=False, usePatches=False, data_augmentation=False):

    training_model_options = net_config['training_model_options']
    training_compile_options = net_config['training_compile_options']
    training_fit_options = net_config['training_fit_options']
    lrate = training_compile_options['initial_lr']

    model = unet.UNET(training_model_options['patch_height'], training_model_options['patch_width'], training_model_options['input_channels'], dataset_config['num_clases_exp' + str(exp_number)] + 1, "channels_last", batchnorm=True, enc_conv_kernel_size=net_config["training_model_options"]["enc_conv_filter_size"], enc_depth=net_config["training_model_options"]["encoder_depth"], n_filters=net_config["training_model_options"]["initial_number_of_filters"], dropout=0.5, implicitNorm=implicit_norm, inference=False, dataAugmentation=data_augmentation)

    gradient_decay_factor = training_compile_options['gradient_decay_factor']
    squared_gradient_decay_factor = training_compile_options['squared_gradient_decay_factor']
    optimizer_fcn = Adam(lrate, gradient_decay_factor, squared_gradient_decay_factor)

    loss_fcn = SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer_fcn, loss=loss_fcn, metrics=[ CustomSparseCategoricalAccuracy(), CustomSparseCategoricalPrecision(), CustomSparseCategoricalIoU()], weighted_metrics=[CustomSparseCategoricalIoU()])

    # Then run the training process
    train_dir = dataset_config['train_dir']
    val_dir = dataset_config['val_dir']
    data_folder = dataset_config['data_folder_train_val_calib']
    label_folder = dataset_config['label_folder_train_val_calib' + str(exp_number)]
    weights = np.array(dataset_config['weights' + str(exp_number)])
    train_batch_size = training_fit_options['train_batch_size']
    val_batch_size = training_fit_options['val_batch_size']
    normMaxValue = dataset_config["maximo_global"]
    normMinValue = dataset_config["minimo_global"]

    train_dataset = Dataset.HSIDriveDataset(train_dir, data_folder, label_folder, weights, normMaxValue, normMinValue, usePatches=usePatches)
    val_dataset = Dataset.HSIDriveDataset(val_dir, data_folder, label_folder, weights, normMaxValue, normMinValue, usePatches=usePatches)
    train_generator = Generator.DataGenerator(train_dataset, dim=(training_model_options["patch_height"], training_model_options["patch_width"]), batch_size=train_batch_size)
    val_generator = Generator.DataGenerator(val_dataset, dim=(training_model_options["patch_height"], training_model_options["patch_width"]), batch_size=val_batch_size)

    # Start the training/finetuning
    epochs = training_fit_options['epochs']
    verbose = training_fit_options['verbose'] #0 = silent, 1 = progress bar, 2 = one line per epoch.
    shuffle = True
    workers = training_fit_options['workers'] #Mikel lo tiene puesto a 0
    use_multiprocessing=False #Mikel lo tiene puesto como False
    max_queue_size = 10 #Mikel lo tiene puesto como 10

    cp_callback = ModelCheckpoint(filepath=float_model, monitor='val_weighted_cat_iou', save_best_only=True, save_weights_only=False, mode='max', save_freq='epoch', verbose=0)
    callbacks = [cp_callback] #Save the model periodically during fit using callbacks

    enable_early_stopping = True
    # Stop training when a monitored metric has stopped improving
    if enable_early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",         # Stop training when `val_loss` is no longer improving
            min_delta=5e-3,             # "no longer improving" being defined as "no better than 1e-2 less"
            patience=80,    # "no longer improving" being further defined as "for at least 2 epochs"
            mode = 'min',               # Stop when the quantity monitored has stopped decreasing
            verbose=1)
    callbacks.append(early_stopping_callback)

    callbacks.append(LearningRateScheduler(lr_time_based_decay, verbose=1))


    model.fit(train_generator, epochs=epochs, verbose=verbose,
              callbacks=callbacks, validation_data=val_generator,
              shuffle=shuffle, use_multiprocessing=use_multiprocessing,
              workers=workers, max_queue_size=max_queue_size)


    # datetime object containing current date and time
    end = datetime.now()
    end_string = end.strftime("%d/%m/%Y %H:%M:%S")
    print("End date and time =", end_string)

    return


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-fm', '--float_model',     type=str, default='./model.h5')
    ap.add_argument('-ex', '--exp_number',      type=int, default=1)
    ap.add_argument('-i',  '--inplicit_norm',   action='store_true')
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
    print(' --float_model        : ', args.float_model)
    print(' --exp_number         : ', args.exp_number)
    print(' --inplicit_norm      : ', args.inplicit_norm)
    print('------------------------------------\n')

    train_model(net_config, dataset_config["versionDASIP"], args.float_model, args.exp_number, args.inplicit_norm)


if __name__ ==  "__main__":
    main()
