import os
from glob import glob
import pickle
from osgeo import gdal
import numpy as np
from time import time
from utils import get_configs, Configuration, ConfigurationIce, make_dirs
import product_info as pinfo
from calibration import calibrate, denoise_crop, calc_angle


class Dataset:
    """
    The Dataset class contains the input data (x) and the label data (y)
    needed to train a machine learning model.
    Attributes:
        x: A numpy array containing the input data.
        x_trn: A numpy array containing a subset of x used for training.
        x_val: A numpy array containing a subset of x used for validation.
        x_tst: A numpy array containing a subset of x used for testing.
        _x_normed: A boolean flag used to indicate whether or not x has
            been z-score normalized (True) or not (False).
        x_trn_mean: A numpy array containing the mean for each feature in
            x_trn among all samples in x_trn.
        x_trn_std: A numpy array containing the standard deviation for
            each feature in x_trn among all samples in x_trn.
        y: A 1:1 numpy array containing the labelled data for each sample
            in x.
        y_trn: A numpy array containing a subset of y used for training,
            which corresponds to x_trn.
        y_val: A numpy array containing a subset of y used for
            validation, which corresponds to x_val.
        y_tst: A numpy array containing a subset of y used for testing,
            which corresponds to x_tst.
    """
    def __init__(self):
        self.x = np.array([])
        self.x_trn = np.array([])
        self.x_val = np.array([])
        self.x_tst = np.array([])
        self._x_normed = False
        self.x_trn_mean = np.array([])
        self.x_trn_std = np.array([])
        self.y = np.array([])
        self.y_trn = np.array([])
        self.y_val = np.array([])
        self.y_tst = np.array([])

    def build(self, c: Configuration):
        """
        To be defined in Dataset subclass for specific project. Thought of
        making this general Dataset class to be used in all 3 of my projects
        (soil moisture estimation, sea ice classification, grain bin
        parameterization) and having DatasetSoil, DatasetIce and
        DatasetGrain subclasses.
        :param c: Configuration
        """
        pass

    def split(self, p_trn=0.8, p_val=0.1):
        """
        Split x and y into training, validation and testing sets.
        :param p_trn: Percentage of dataset to be used for training.
        :param p_val: Percentage of dataset to be used for validation.
            Testing set size is inferred.
        :return:
        """
        # Calculate number of train, validation, and test samples.
        n_samples = self.x.shape[0]
        n_trn = int(p_trn * n_samples)
        n_val = int(p_val * n_samples)
        n_tst = n_samples - n_trn - n_val
        assert n_samples == n_trn + n_val + n_tst

        # Split input data
        self.x_trn = self.x[:n_trn]
        self.x_val = self.x[n_trn: n_trn + n_val]
        self.x_tst = self.x[-n_tst:]
        assert self.x_trn.shape[0] + self.x_val.shape[0] + \
               self.x_tst.shape[0] == n_samples

        # Split output data
        self.y_trn = self.y[:n_trn]
        self.y_val = self.y[n_trn: n_trn + n_val]
        self.y_tst = self.y[-n_tst:]
        assert self.y_trn.shape[0] + self.y_val.shape[0] + \
               self.y_tst.shape[0] == n_samples

    def normalize(self):
        """
        Z-score normalize dataset.
        :return:
        """
        if self._x_normed is False:
            if self.x_trn_mean.size == 0 and self.x_trn_std.size == 0:
                tmp_mean = np.mean(self.x_trn, axis=-1)
                tmp_mean = np.mean(tmp_mean, axis=-1)
                self.x_trn_mean = np.mean(tmp_mean, axis=0).reshape((-1, 1, 1))
                tmp_std = np.std(self.x_trn, axis=-1)
                tmp_std = np.std(tmp_std, axis=-1)
                self.x_trn_std = np.std(tmp_std, axis=0).reshape((-1, 1, 1))
            self.x_trn = norm(self.x_trn, self.x_trn_mean, self.x_trn_std)
            self.x_val = norm(self.x_val, self.x_trn_mean, self.x_trn_std)
            self.x_tst = norm(self.x_tst, self.x_trn_mean, self.x_trn_std)
            self._x_normed = True
        else:
            print('Dataset is already normalized.')

    def inverse_normalize(self):
        """
        Inverse the z-score normalization.
        """
        if self._x_normed is True:
            self.x_trn = inv_norm(self.x_trn, self.x_trn_mean, self.x_trn_std)
            self.x_val = inv_norm(self.x_val, self.x_trn_mean, self.x_trn_std)
            self.x_tst = inv_norm(self.x_tst, self.x_trn_mean, self.x_trn_std)
            self._x_normed = False
        else:
            print('Dataset has not been normalized yet.')

    def x_is_normed(self):
        """
        Getter function to return protected attributed _x_normed used to
        denote if the dataset is normalized.
        :return: _x_normed
        """
        return self._x_normed is True

    def shuffle(self, seed=0):
        """
        Randomly shuffle x and y with a given seed (default 0).
        :param seed: Integer used to seed the random number generator
        """
        self.x = shuffle(self.x, seed)
        self.y = shuffle(self.y, seed)

    def save(self, file_path):
        """
        Pickle the Dataset.
        :param file_path: Path where to save Dataset.
        """
        path = '/'.join(file_path.split('/')[:-1])
        # Make sure path for output exists, else create.
        _ = make_dirs(path)
        if os.path.isfile(file_path):
            print(f'File {file_path} already exists. File was not overwritten.')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, protocol=4)

    def load(self, file_path):
        """
        Unpickle a Dataset.
        :param file_path: Path of a saved Dataset to load.
        :return:
        """
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        return self


class DatasetIce(Dataset):
    """
    The DatasetIce class contains the input data (x) and the label
    data (y) needed to train a machine learning model for the sea ice
    classification project.
    Attributes:
        x: Shape = (n_samples, n_channels, n_tx, n_rx).
            n_channels is the number of frequencies used to interrogate
            the medium within the grain bin.
            n_tx is the number of transmitters in the antenna array
            installed in a grain bin. The antenna array normally has one
            antenna configured to transmit, and the remaining antennas are
            configured to receive.
            n_rx = n_tx - 1.
        y: Shape = (n_samples, n_parameters).
            n_parameters is defined based on the length of the parameters
            list in ConfigurationGrain.
    """
    def __init__(self):
        super().__init__()
        self.t = np.array([])
        self.t_trn = np.array([])
        self.t_val = np.array([])
        self.t_tst = np.array([])

    def gather(self, file_groups, c: ConfigurationIce):
        """
        Gather crops pointed to by paths in file_groups into x.
        :param file_groups: Nested lists of crop paths grouped per source image
        :param c: ConfigurationIce object
        """
        t_start = time()
        n_done = 0
        n_files = np.sum([group.shape[0] for group in file_groups])
        n_digits = int(np.log10(n_files)) + 1
        img_set, angle_set, title_set = [], [], []
        for group in file_groups:
            dir_tokens = group[0][0].split('/')[4].replace('.tif', '').split(
                '_')
            sar_root_name = '_'.join(dir_tokens[:-1] + ['HH_HV_SGF'])
            path_raw_sar = os.path.join('./Dataset/Raw_SAR', sar_root_name)
            path_pif = os.path.join(path_raw_sar, 'product.xml')
            path_beta = os.path.join(path_raw_sar, 'lutBeta.xml')
            path_sigma = os.path.join(path_raw_sar, 'lutSigma.xml')

            # Get auxiliary data for pre-processing steps
            raw_rows, raw_cols = pinfo.get_dims(path_pif)
            geo_height = pinfo.get_geo_height(path_pif)
            rf = pinfo.get_rational_function(path_pif)
            noise = pinfo.get_noise(path_pif)
            beta = pinfo.get_look_up_table(path_beta)
            sigma = pinfo.get_look_up_table(path_sigma)

            # Iterate over lists of crop paths to create samples
            for i, file_crop_list in enumerate(group):
                img = []
                # Add polarizations as channels in an image
                for crop_pol_file in file_crop_list:
                    ds_crop = gdal.Open(crop_pol_file)
                    crop = ds_crop.GetRasterBand(1).ReadAsArray()
                    img.append(np.expand_dims(crop, axis=0))
                img = np.concatenate(img, axis=0)

                # Give the sample a title/name for easy reference/verification
                fname = file_crop_list[0].split('/')[-1]
                title_tokens = fname.replace('.tif', '').split('_')
                title = '_'.join([title_tokens[0]] + title_tokens[4:])
                title = title.replace(c.pol[0], '+'.join(c.pol))
                title_set.append(title)

                # Perform pre-processing
                lat, lon = np.float64(title_tokens[-2:])
                x_ind, y_ind = rf.lat_lon_index(lat, lon, geo_height)
                beta_crop = beta[x_ind - 50: x_ind + 50]
                sigma_crop = sigma[x_ind - 50: x_ind + 50]
                if c.pre_processing == 'raw':
                    pass  # Do nothing
                elif 'cal' in c.pre_processing:
                    img = calibrate(img, sigma_crop)
                    if 'denoise' in c.pre_processing:
                        # Denoise
                        img = denoise_crop(img, x_ind, noise, raw_cols, 0)

                # Always add a channel for the angle
                # Note: Addition of angle channel was found to be detrimental
                # to results. For this reason, self.x does not get updated
                # with angle set.
                img_angle = calc_angle(beta_crop, sigma_crop, img.shape[1:])
                angle_set.append(np.expand_dims(img_angle, axis=0))
                img_set.append(np.expand_dims(img, axis=0))
                n_done += 1
                if (n_done % int(n_files / 100) == 0) or (n_done == n_files):
                    print(f'\rPacking image progress: '
                          f'{n_done:{n_digits}}/{n_files} '
                          f'({n_done / n_files * 100:3.0f}%)', end='')
        img_set = np.concatenate(img_set, axis=0)
        angle_set = np.concatenate(angle_set, axis=0)

        # Set appropriate channels to x per configuration
        n_samples, n_ch, height, width = img_set.shape
        # self.x = np.empty((n_samples, len(c.pol) + 1, height, width))
        self.x = np.empty((n_samples, len(c.pol), height, width))
        for i, pol in enumerate(c.pol):
            if pol == 'HH':
                self.x[:, i, :] = img_set[:, 0, :]
            elif pol == 'HV':
                if n_ch == 1:
                    self.x[:, i, :] = img_set[:, 0, :]
                else:
                    self.x[:, i, :] = img_set[:, 1, :]
            elif pol == 'HH_DIV_HV':
                np.seterr(divide='ignore', invalid='ignore')
                hh_div_hv = img_set[:, 0, :] / img_set[:, 1, :]
                hh_div_hv[np.isnan(hh_div_hv)] = 0
                hh_div_hv[np.isinf(hh_div_hv)] = 0
                self.x[:, i, :] = hh_div_hv
            elif pol == 'HV_DIV_HH':
                np.seterr(divide='ignore', invalid='ignore')
                hv_div_hh = img_set[:, 1, :] / img_set[:, 0, :]
                hv_div_hh[np.isnan(hv_div_hh)] = 0
                hv_div_hh[np.isinf(hv_div_hh)] = 0
                self.x[:, i, :] = hv_div_hh
        # self.x[:, -1, :] = angle_set
        self.x = np.float32(self.x)
        self.t = np.array(title_set)
        t_stop = time()
        print(f'\nTotal dataset build time: {(t_stop - t_start) / 60} minutes.')

    def build(self, c: ConfigurationIce):
        """
        Build DatasetIce by populating x, y, and t. Also perform shuffling,
        create trn, val, and tst split, and normalize.
        :param c: ConfigurationIce object
        """
        path_sar = './Dataset/Labelled_SAR_Data/'
        path_package = './Dataset/Packaged_Dataset/'
        ds_code = c.name.split('_')[0]
        ds_name = ds_code + '.pickle'
        path_ds = os.path.join(path_package, ds_name)

        if os.path.isfile(path_ds):
            print(f'Loading dataset found at {path_ds}')
            t0 = time()
            self.load(path_ds)
            t1 = time()
            print(f'Time to load: {(t1 - t0) / 60} minutes.')
        else:
            print(f'No dataset found at {path_ds}\n'
                  f'Building dataset.')
            path_f = check_input_path(path_sar, ds_code)
            with open(path_f, 'rb') as f:
                # dict containing dataset (ds) of paths and labels (pl)
                ds_pl = pickle.load(f)

            # Group crops based on their parent directory (source image) for
            # more efficient opening of auxiliary files in gather() such as
            # product.xml, lutBeta.xml, and lutSigma.xml for example.
            sar_paths = ds_pl['sar_paths']
            dirs = np.char.rsplit(sar_paths[:, 0], sep='/')
            dirs = np.vstack([np.array(row) for row in dirs])
            img_dirs = dirs[:, 4]
            _, n_samples_per_dir = np.unique(img_dirs, return_counts=True)
            groups_idx = np.cumsum(n_samples_per_dir)[:-1]
            groups = np.split(sar_paths, groups_idx)
            self.gather(groups, c)
            self.y = ds_pl['labels']

            # Reduce Water Class Dataset. (Balanced dataset preferred)
            # unique_labels = np.unique(np.squeeze(self.y), return_counts=True)
            # n_water = unique_labels[1][0]
            # n_max_ice = np.max(unique_labels[1][1:])
            # if n_water > (1.5 * n_max_ice):
            #     water_indices = np.where(self.y == c.code_to_label['OW'])[0]
            #     ice_indices = np.where(self.y != c.code_to_label['OW'])[0]
            #     np.random.seed(0)
            #     water_keep = np.random.choice(
            #         water_indices, n_max_ice, replace=False)
            #     keep_indices = np.sort(np.append(water_keep, ice_indices))
            #     self.x = self.x[keep_indices]
            #     self.y = self.y[keep_indices]
            #     self.t = self.t[keep_indices]

            # Balance Dataset
            unique_labels = np.unique(np.squeeze(self.y), return_counts=True)
            n_min_class = np.min(unique_labels[1])
            for i, class_type in enumerate(unique_labels[0]):
                if unique_labels[1][i] != n_min_class:
                    class_indices = np.where(self.y == class_type)[0]
                    other_indices = np.where(self.y != class_type)[0]
                    np.random.seed(0)
                    class_keep = np.random.choice(
                        class_indices, n_min_class, replace=False)
                    keep_indices = np.sort(np.append(class_keep, other_indices))
                    self.x = self.x[keep_indices]
                    self.y = self.y[keep_indices]
                    self.t = self.t[keep_indices]

            print(f'Saving Dataset in {path_ds}')
            self.save(path_ds)
        t0 = time()
        self.shuffle()
        self.split()
        self.normalize()
        t1 = time()
        print(f'Time to shuffle, split and normalize: '
              f'{(t1 - t0) / 60} minutes.')

    def split(self, p_trn=0.8, p_val=0.1):
        super().split(p_trn, p_val)
        # Calculate number of train, validation, and test samples.
        n_samples = self.x.shape[0]
        n_trn = self.x_trn.shape[0]
        n_val = self.x_val.shape[0]
        n_tst = self.x_tst.shape[0]

        # Split input data
        self.t_trn = self.t[:n_trn]
        self.t_val = self.t[n_trn: n_trn + n_val]
        self.t_tst = self.t[-n_tst:]
        assert self.t_trn.shape[0] + self.t_val.shape[0] + \
               self.t_tst.shape[0] == n_samples

    def shuffle(self, seed=0):
        super().shuffle(seed)
        self.t = shuffle(self.t, seed)

    def stats(self):
        unique_labels = np.unique(np.squeeze(self.y), return_counts=True)
        dist = unique_labels[1] / np.sum(unique_labels[1])
        print('Dataset Distribution:')
        print(f'Classes {unique_labels[0]}')
        print(f'Counts  {unique_labels[1]}')
        with np.printoptions(formatter={'float': '{:4.2f}'.format}):
            print(f'Percent {dist * 100}')


def norm(a: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (a - mean) / std


def inv_norm(a: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return a * std + mean


def shuffle(a: np.ndarray, seed=0):
    np.random.seed(seed)
    np.random.shuffle(a)
    return a


def check_input_path(path_d, name_f):
    path_f = os.path.join(path_d, name_f[:-1] + 'X' + '.pickle')
    if os.path.isfile(path_f):
        print(f'Using the SAR crop paths and labels in {path_f} to build the '
              f'dataset.')
    elif len(glob(os.path.join(path_d, name_f[:-1] + '?.pickle'))) > 0:
        print(f'{path_f} does not exist.')
        path_f = glob(os.path.join(path_d, name_f[:-1] + '?.pickle'))[0]
        print(f'Using the SAR crop paths and labels in {path_f}\n'
              f'to build the dataset instead since the crop paths and '
              f'labels are independent of the pre-processing type.\n'
              f'The pre-processing type only takes effect while building '
              f'the dataset.')
    else:
        print(f'{path_f} does not exist and no other variant '
              f'{os.path.join(path_d, name_f[:-1] + "?.pickle")} exists.')
        exit()
    return path_f


def main():
    config: ConfigurationIce = get_configs()
    ds = DatasetIce()
    ds.build(config)
    ds.stats()
    return


if __name__ == '__main__':
    main()
    print('Program Terminated')
