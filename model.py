import os
import sys
import time
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import \
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils import make_dirs, get_configs, ConfigurationIce, Logger
from dataset import DatasetIce
import metrics
import networks
from analysis_tools import plot_history, plot_img, \
    print_label_acc, print_monthly_acc, print_label_acc_per_month, \
    print_label_num_per_month, unique_titles, crops_to_scene


class ModelIce(object):
    def __init__(self, c: ConfigurationIce):
        self.model = None
        self.path_results = os.path.join('./Results', 'results_' + c.name)
        self.path_kmodel = os.path.join(self.path_results, 'model_keras.h5')
        self.func_map = make_object_map(c)
        self.x_trn_mean = np.array([])
        self.x_trn_std = np.array([])
        self.y_trn_mean = np.array([])
        self.y_trn_std = np.array([])
        self.trn_time = 0

    def train(self, ds: DatasetIce, c: ConfigurationIce):
        # Make results directory
        _ = make_dirs(self.path_results)

        input_shape = ds.x_trn.shape[1:]
        trn_set = ds.x_trn
        val_set = ds.x_val
        trn_labels_cat = to_categorical(ds.y_trn)
        val_labels_cat = to_categorical(ds.y_val)
        num_labels = trn_labels_cat.shape[1]

        # Store the training mean and standard deviation of the dataset used
        # to train the model.
        self.x_trn_mean = ds.x_trn_mean
        self.x_trn_std = ds.x_trn_std

        # Get model object
        model_func = getattr(networks, c.model)
        if c.model == 'unet':
            pad = int((np.ceil(input_shape[0] / 16) * 16 - input_shape[0]) / 2)
            trn_set = np.pad(ds.x_trn,
                               [(0, 0), (pad, pad), (pad, pad), (0, 0)])
            val_set = np.pad(ds.x_val,
                              [(0, 0), (pad, pad), (pad, pad), (0, 0)])
            input_shape = trn_set.shape[1:]

        model = model_func(input_shape, num_labels)
        model.summary()
        model.compile(loss=getattr(metrics, c.loss),
                      optimizer=optimizers.Adam(),
                      metrics=[getattr(metrics, m) for m in c.metric])
        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=c.es_patience,
                          verbose=1),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=c.lr_patience,
                              min_lr=0.00001,
                              verbose=1),
            ModelCheckpoint(filepath=self.path_kmodel,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True)
        ]
        t0 = time.time()
        history = model.fit(trn_set,
                            trn_labels_cat,
                            batch_size=c.batch_size,
                            epochs=c.epochs,
                            shuffle=True,
                            validation_data=(val_set,
                                             val_labels_cat),
                            callbacks=callbacks)
        t1 = time.time()
        self.trn_time = (t1 - t0) / 60
        print(f'Time to train: {self.trn_time} minutes.')

        # Save and plot the history
        hist_dict = history.history
        for key, val in hist_dict.items():
            if isinstance(hist_dict[key][0], np.float32):
                hist_dict[key] = [np.float64(x) for x in hist_dict[key]]
        with open(os.path.join(self.path_results, 'history.json'), 'wt') as f:
            json.dump(hist_dict, f, indent=4)
        plot_history(hist_dict, self.path_results)

        model.load_weights(self.path_kmodel)
        return model

    def test(self, ds: DatasetIce, c: ConfigurationIce):
        # Set up output logging
        logfile = os.path.join(self.path_results, 'console_output.txt')
        sys.stdout = Logger(logfile)

        trn_set = ds.x_trn
        tst_set = ds.x_tst
        trn_labels = np.squeeze(ds.y_trn)
        tst_labels = np.squeeze(ds.y_tst)
        trn_labels_cat = to_categorical(ds.y_trn)
        tst_labels_cat = to_categorical(ds.y_tst)
        trn_titles = ds.t_trn
        tst_titles = ds.t_tst
        num_labels = trn_labels_cat.shape[1]

        if c.model == 'unet':
            pad = int((np.ceil(trn_set.shape[2] / 16) * 16 -
                       trn_set.shape[2]) / 2)
            trn_set = np.pad(trn_set, [(0, 0), (0, 0), (pad, pad), (pad, pad)])
            tst_set = np.pad(tst_set, [(0, 0), (0, 0), (pad, pad), (pad, pad)])

        print('Model Summary:')
        self.model.summary()
        print()
        print(f'Time to train: {self.trn_time} minutes.')
        print()
        print('Evaluating the model on the training and testing sets.')
        train_results = self.model.evaluate(trn_set,
                                            trn_labels_cat,
                                            batch_size=c.batch_size)
        test_results = self.model.evaluate(tst_set,
                                           tst_labels_cat,
                                           batch_size=c.batch_size)

        # Make predictions
        trn_pred = self.model.predict(trn_set)
        trn_pred_classes = np.argmax(trn_pred, axis=1)
        trn_rw = trn_labels == trn_pred_classes  # Right/wrong pred on train
        tst_pred = self.model.predict(tst_set)
        tst_pred_classes = np.argmax(tst_pred, axis=1)
        tst_rw = tst_labels == tst_pred_classes  # Right/wrong pred on test

        # Calculate the accuracy for the ice labels only.
        trn_ice_idx = np.where(trn_labels >= c.code_to_label['1'])
        trn_ice_rw = trn_rw[trn_ice_idx]
        trn_ice_acc = np.sum(trn_ice_rw) / trn_ice_rw.size

        tst_ice_idx = np.where(tst_labels >= c.code_to_label['1'])
        tst_ice_rw = tst_rw[tst_ice_idx]
        tst_ice_acc = np.sum(tst_ice_rw) / tst_ice_rw.size

        # Get unique months.
        title_months = np.char.split(ds.t, sep='_')
        title_months = np.vstack(
            [np.array(row)[2][:-2] for row in title_months])
        title_months = np.unique(title_months)

        # Training Accuracy Stats
        print('Training Results:')
        print(f'Loss:   Categorical Cross-entropy = {train_results[0]:.4}')
        print(f'Metric:                  Accuracy = {train_results[1]:.4}')
        print(f'Metric:              Ice Accuracy = {trn_ice_acc:.4}')
        print()
        print('Training Accuracy Per Class:')
        print_label_acc(np.unique(ds.y), trn_labels, trn_rw)
        print()
        print('Training Accuracy Monthly Breakdown:')
        print_monthly_acc(title_months, trn_titles, trn_rw, 'training')
        print()
        print('Training Samples Monthly Breakdown Per Class:')
        print_label_num_per_month(np.unique(ds.y), trn_labels, title_months,
                                  trn_titles)
        print()
        print('Training Accuracy Monthly Breakdown Per Class:')
        print_label_acc_per_month(np.unique(ds.y), trn_labels, trn_rw,
                                  title_months, trn_titles)
        print()

        # Testing Accuracy Stats
        print('Testing Results:')
        print(f'Loss:   Categorical Cross-entropy = {test_results[0]:.4}')
        print(f'Metric:                  Accuracy = {test_results[1]:.4}')
        print(f'Metric:              Ice Accuracy = {tst_ice_acc:.4}')
        print()
        print('Testing Accuracy Per Class:')
        print_label_acc(np.unique(ds.y), tst_labels, tst_rw)
        print()
        print('Testing Accuracy Monthly Breakdown:')
        print_monthly_acc(title_months, tst_titles, tst_rw, 'testing')
        print()
        print('Testing Samples Monthly Breakdown Per Class:')
        print_label_num_per_month(np.unique(ds.y), tst_labels, title_months,
                                  tst_titles)
        print()
        print('Testing Accuracy Monthly Breakdown Per Class:')
        print_label_acc_per_month(np.unique(ds.y), tst_labels, tst_rw,
                                  title_months, tst_titles)
        print()

        trn_unique_counts = np.unique(trn_labels, return_counts=True)
        trn_dist = trn_unique_counts[1] / np.sum(trn_unique_counts[1])
        tst_unique_counts = np.unique(tst_labels, return_counts=True)
        tst_dist = tst_unique_counts[1] / np.sum(tst_unique_counts[1])

        # Create confusion matrices.
        train_confusion = np.zeros((num_labels, num_labels), dtype=int)
        for pred, label in zip(trn_pred_classes, trn_labels):
            train_confusion[int(pred), int(label)] += 1

        test_confusion = np.zeros((num_labels, num_labels), dtype=int)
        for pred, label in zip(tst_pred_classes, tst_labels):
            test_confusion[int(pred), int(label)] += 1

        print('Train Distribution:')
        print(f'Classes {trn_unique_counts[0]}')
        print(f'Counts  {trn_unique_counts[1]}')
        with np.printoptions(formatter={'float': '{:4.2f}'.format}):
            print(f'Percent {trn_dist * 100}')
        print()
        print('Training confusion matrix:')
        print(train_confusion)
        print()

        print('Test Distribution:')
        print(f'Classes {tst_unique_counts[0]}')
        print(f'Counts  {tst_unique_counts[1]}')
        with np.printoptions(formatter={'float': '{:4.2f}'.format}):
            print(f'Percent {tst_dist * 100}')
        print()
        print('Testing confusion matrix:')
        print(test_confusion)
        print()

        # Get paths to SAR images
        sar_img_files = []
        path_data_batch = os.path.join(
            './Dataset/Batches',
            config.data_batch + '.json')
        with open(path_data_batch, 'rb') as f:
            batch_paths = json.load(f)
        sar_paths = batch_paths['SAR']

        polarizations = config.pol
        if 'HH_DIV_HV' in config.pol or 'HV_DIV_HH' in config.pol:
            polarizations = ['HH', 'HV']

        for pol in polarizations:
            pol_files = []
            for i, p in enumerate(sar_paths):
                raw_sar_path = p + f'_HH_HV_SGF/imagery_{pol}.tif'
                if os.path.isfile(raw_sar_path):
                    pol_files.append(raw_sar_path)
                else:
                    raise FileNotFoundError(
                        f'No such file: {raw_sar_path}.\n'
                        f'Invalid polarization in configuration or '
                        f'imagery_{pol}.tif must be created.'
                    )
            sar_img_files.append(pol_files)
        sar_img_files = np.column_stack(sar_img_files)

        trn_titles = np.char.replace(trn_titles, '+'.join(c.pol), 'HH')
        tst_titles = np.char.replace(tst_titles, '+'.join(c.pol), 'HH')

        sar_scenes = unique_titles(tst_titles)
        trn_scenes_indices = crops_to_scene(trn_titles, sar_scenes)
        tst_scenes_indices = crops_to_scene(tst_titles, sar_scenes)
        # Alternative
        # sar_scenes = np.char.split(tst_titles, sep='_')
        # sar_scenes = np.vstack(['_'.join(np.array(row)[:4]) for
        #                        row in sar_scenes])
        # tst_scenes_indices = []
        # for e in np.unique(sar_scenes):
        #     tst_scenes_indices.append(np.where(np.in1d(sar_scenes, e))[0])

        # Create heuristic for sorting which SAR scenes to visualize.
        scene_label_counts = np.zeros((len(sar_scenes), len(c.label_names)))
        for i, ind_list in enumerate(tst_scenes_indices):
            scene_labels = tst_labels[np.array(ind_list)]
            classes, counts = np.unique(scene_labels, return_counts=True)
            scene_label_counts[i, np.int32(classes)] = counts
        sorted_scene_label_counts = np.sort(scene_label_counts, axis=1)
        scene_label_ratios = (sorted_scene_label_counts[:, :-1].T /
                              sorted_scene_label_counts[:, -1]).T
        h1 = np.sum(scene_label_ratios, axis=1)
        h_norm = h1 / np.max(h1)
        h2 = np.max(scene_label_counts, axis=1) * h_norm
        sort_idx = np.argsort(h2)[::-1]

        # Alternate heuristic.
        # lengths = [len(a) for a in indices]
        # sort_idx = np.argsort(lengths)[::-1]just

        # Apply heuristic.
        sorted_sar_scenes = np.array(sar_scenes)[sort_idx]
        sorted_trn_scenes_indices = np.array(trn_scenes_indices)[sort_idx]
        sorted_tst_scenes_indices = np.array(tst_scenes_indices)[sort_idx]

        # Visualize predictions.
        plot_img(trn_labels, trn_pred_classes, trn_titles, sar_img_files,
                 sorted_sar_scenes, sorted_trn_scenes_indices, 'train',
                 c.pre_processing, self.path_results)

        plot_img(tst_labels, tst_pred_classes, tst_titles, sar_img_files,
                 sorted_sar_scenes, sorted_tst_scenes_indices, 'test',
                 c.pre_processing, self.path_results)

        sys.stdout.close()
        print("Analysis tests for synthetic data terminated.")

    def save(self, file_path):
        path = '/'.join(file_path.split('/')[:-1])
        # Make sure path for output exists, else create.
        _ = make_dirs(path)
        if os.path.isfile(file_path):
            print(f'File {file_path} already exists. File was not overwritten.')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, protocol=4)

    def load(self, file_path):
        # Load ModelIce object
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)

        # Load keras model
        if os.path.isfile(self.path_kmodel):
            self.model = models.load_model(self.path_kmodel,
                                           custom_objects=self.func_map)
        else:
            raise FileNotFoundError(
                f'The Keras model at {self.path_kmodel} does not exist. '
                f'This occurs when the model has not been trained yet or the '
                f'incorrect path_results was given in the configuration file.'
            )


def run_model(ds: DatasetIce, c: ConfigurationIce):
    # Disable the warning:
    # "Your CPU supports instructions that this TensorFlow binary was not
    # compiled to use: AVX2 FMA"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Limit TensorFlow's automatic reservation of the entire GPU memory.
    cp = tf.compat.v1.ConfigProto()
    cp.gpu_options.allow_growth = True
    k.set_session(tf.compat.v1.Session(config=cp))
    k.set_floatx('float32')
    k.set_image_data_format('channels_first')

    imodel = ModelIce(c)
    model_ice_path = os.path.join(imodel.path_results, 'model_ice.pickle')
    if os.path.isfile(model_ice_path):
        print(f'Loading ice model found at {model_ice_path}')
        imodel.load(model_ice_path)
    else:
        path_package = './Dataset/Packaged_Dataset/'
        ds_code = c.name.split('_')[0]
        ds_name = ds_code + '.pickle'
        path_ds = os.path.join(path_package, ds_name)
        print(f'No ice model found at {model_ice_path}\n'
              f'Training a model on the dataset at {path_ds}')
        kmodel = imodel.train(ds, c)
        # Ice model mean and std attributes are set in train and this
        # is what we want saved and kept together. The Keras model is
        # saved separately by Keras.
        print(f'Storing the ice model in {model_ice_path}')
        imodel.save(model_ice_path)
        imodel.model = kmodel
    imodel.test(ds, c)
    return imodel


def make_object_map(c: ConfigurationIce):
    func_map = {c.model: getattr(networks, c.model),
                c.loss: getattr(metrics, c.loss)}
    for m in c.metric:
        func_map[m] = getattr(metrics, str(m))
    return func_map


if __name__ == '__main__':
    config = get_configs()
    dataset = DatasetIce()
    dataset.build(config)
    ice_model = run_model(dataset, config)
    print('Program Terminated')
