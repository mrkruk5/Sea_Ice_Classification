import os
import json
import pickle
import numpy as np
import copy
from functools import reduce
from osgeo import gdal
from datetime import datetime, timedelta
from calibration import create_crop_GCPs
from utils import get_configs, ConfigurationIce, make_dirs
import product_info as pinfo


def find_dex(sar_file, dex_list, dex_type):
    """
    Find the corresponding DEX file for a given sar_file.
    :param sar_file: path to a SAR file
    :param dex_list: list of paths to DEX files
    :param dex_type: DEXA or DEXI
    :return: data contained in the corresponding DEX file if found, None o/w
    """
    dex_data = None
    # Perform linear search through dex_list for match
    for dex_file in dex_list:
        if dex_type == 'DEXA':
            # Extract date (YYYYMMDD) from name of sar_file
            date = sar_file.split('/')[-2].split('_')[5]
            if date in dex_file:
                with open(dex_file, 'rb') as ff:
                    dex_data = pickle.load(ff)
                return dex_data
        else:
            # For image files (DEXI), a more specific matching criteria is
            # used by also matching the time of acquisition
            sar_timestamp = ''.join(sar_file.split('/')[-2].split('_')[5:7])
            sar_date = datetime.strptime(sar_timestamp, '%Y%m%d%H%M%S')
            dex_timestamp = dex_file.split('/')[-1].split('_')[0]
            dex_date = datetime.strptime(dex_timestamp, '%Y%m%d%H%M%S')
            delta = timedelta(minutes=5)  # Somewhat arbitrary
            # Try exact match.
            if sar_timestamp in dex_file:
                with open(dex_file, 'rb') as ff:
                    dex_data = pickle.load(ff)
            # Try time range.
            elif sar_date - delta <= dex_date <= sar_date + delta:
                with open(dex_file, 'rb') as ff:
                    dex_data = pickle.load(ff)
                return dex_data
    return dex_data


def crop_img(sar_file, dex_data, pol, hw):
    """
    Generate crops in SAR scene pointed to by sar_file centered about the lat,
    lon contained in dex_data and maintain the label (also given in
    dex_data) for the generated crops.
    :param sar_file: path to SAR image
    :param dex_data: ndarray of lat, lon, and labels extracted from a DEX
        file which has been processed by DEX_reader.py and label_egg_codes.py.
    :param pol: polarization (HH or HV)
    :param hw: half-width
    :return: count of crops created and list of paths to crops to use and
        their labels
    """
    PATH_CROP = os.path.join(
        './Dataset/Cropped_SAR_Data/', pol)
    pol = sar_file.split('/')[-1].split('.')[0].split('_')[-1]
    path_sar = '/'.join(sar_file.split('/')[:-1])
    path_pif = path_sar + '/product.xml'  # Product Information File

    crops_in_img = []
    ds_sar = gdal.Open(sar_file)
    sar_img = ds_sar.GetRasterBand(1).ReadAsArray()
    rows, cols = sar_img.shape
    sar_extents = pinfo.get_extents(path_pif, rows, cols)
    geo_min = np.min(sar_extents, axis=0)
    geo_max = np.max(sar_extents, axis=0)

    # Warp source image to get geo-transform of warped result.
    options = gdal.WarpOptions(
        format='VRT',
        outputBounds=(*geo_min, *geo_max),
        width=cols,
        height=rows,
        dstSRS='EPSG:4326'
    )
    ds_src_warp = gdal.Warp(
        '',
        path_sar,
        options=options
    )
    geo_transform = ds_src_warp.GetGeoTransform()
    # Create transformer (as seen in test_cropping_with_GDAL_bindings()
    # function in calibration.py). This is needed since Rational functions
    # relate lat, lon to unwarped image coordinates, but for verification
    # purposes, we need coordinates in warped image and corresponding GCPs
    # for crop to render properly in GIS application.
    tx = gdal.Transformer(ds_sar, ds_src_warp, ['METHOD=GCP_TPS'])

    # Filter for dex_data contained within the extents of the SAR image
    dex_filtered = dex_data[
        np.where((geo_min[1] <= dex_data[:, 0]) &
                 (dex_data[:, 0] <= geo_max[1]) &
                 (geo_min[0] <= dex_data[:, 1]) &
                 (dex_data[:, 1] <= geo_max[0]))
    ]

    new_count = 0
    rf = pinfo.get_rational_function(path_pif)
    geo_height = pinfo.get_geo_height(path_pif)
    for lat, lon, label in dex_filtered:
        # Get image coordinates from lat, lon using Rational Function
        x_ind, y_ind = rf.lat_lon_index(lat, lon, geo_height)
        if (0 < (x_ind - hw)) and ((x_ind + hw) < cols) and \
                (0 < (y_ind - hw)) and ((y_ind + hw) < rows):
            crop = sar_img[y_ind - hw: y_ind + hw, x_ind - hw: x_ind + hw]
            # Only use crop if 80% of its pixels are good data (not fill data)
            if np.count_nonzero(crop) > 0.8 * (2 * hw) ** 2:  # 0.8 is arbitrary
                tokens = sar_file.split('/')[-2].split('_')[:-3]
                tokens.append(pol)
                new_dir = os.path.join(PATH_CROP, '_'.join(tokens))
                make_dirs(new_dir)
                name = '_'.join(tokens + [str(lat), str(lon)]) + '.tif'
                name = os.path.join(new_dir, name)
                if os.path.isfile(name):
                    crops_in_img.append([name, label])
                else:
                    new_count += 1
                    # convert Rational Function x_ind and y_ind to warped image
                    # coordinates using tx and create GCPs for
                    # crop in warped image for it to display properly in GIS
                    # application.
                    gcps = create_crop_GCPs(tx, geo_transform, crop.shape,
                                            x_ind - hw, y_ind - hw,
                                            geo_height, 2)
                    # gdal_translate's x-axis is left-to-right and its y-axis
                    # is top-to-bottom from the top left of the image.
                    # Setting of proper GCPs is crucial for proper rendering.
                    options = gdal.TranslateOptions(
                        srcWin=[x_ind - hw, y_ind - hw, 2 * hw, 2 * hw],
                        GCPs=gcps)
                    ds_crop = gdal.Translate(name, ds_sar, options=options)
                    ds_crop.FlushCache()  # Save to disk.
                    crops_in_img.append([name, label])
    return new_count, crops_in_img


def generate_crops(sar_list, dex_list, hw, dex_type, pol):
    """
    For each SAR path provided in sar_list, find the corresponding DEX file
    within dex_list,
    :param sar_list: list of SAR paths
    :param dex_list: list of DEX paths
    :param hw: half-width of crop
    :param dex_type: type of DEX file to use (DEXA or DEXI)
    :param pol: SAR polarization to use (HH or HV)
    :return: list of paths to crops generated
    """
    crops_in_dataset = []
    for i, sar_file in enumerate(sar_list):
        dex_data = find_dex(sar_file, dex_list, dex_type)
        if dex_data is not None:
            count, crops_in_img = crop_img(sar_file, dex_data, pol, hw)
            crops_in_dataset.extend(crops_in_img)
            print(f'There were {count} new crops generated from '
                  f'{sar_file}. ({i + 1}/{len(sar_list)})')
        else:
            print(f'No DEX file found for {sar_file}. '
                  f'({i + 1}/{len(sar_list)})')
    return np.array(crops_in_dataset)


def intersection(path_set, label_set, pols):
    """
    Find intersection between sub-lists in path_set that contain paths to
    crops for different polarizations. This function ensures that the same
    crops (same lat, lon) for different polarizations are linked on the
    same row with the correct label in the output.
    :param path_set: nested list of crop paths grouped by polarization
    :param label_set: nested list of labels
    :param pols: list of polarizations
    :return: ndarrays of crop paths and labels
    """
    n_ch = len(path_set)  # Number of channels.
    ch_paths_strip = copy.deepcopy(path_set)  # Stripped paths for each ch.
    for i, (paths, pol) in enumerate(zip(path_set, pols)):
        stripped = np.char.replace(paths, pol, '')
        ch_paths_strip[i] = stripped
    # Use reduce to find common elements across more than 2 lists
    common = reduce(np.intersect1d, ch_paths_strip)
    # Assign memory for output
    paths_com = np.empty((common.size, n_ch), dtype=path_set[0].dtype)
    label_com = np.empty((common.size, n_ch))
    for i, (paths, paths_strip) in enumerate(zip(path_set, ch_paths_strip)):
        # Update output according to index where elements in paths_strip are
        # in common
        idx = np.where(np.in1d(paths_strip, common))
        paths_com[:, i] = np.squeeze(paths[idx])
        label_com[:, i] = label_set[i][idx]

    label_com = np.unique(label_com, axis=1)
    # Check that cropping worked the same across all polarizations.
    # If each column in a row is not the same, np.unique will not return 1
    # element in each row (it will return an ordered set of unique values in
    # the row).
    assert (label_com.shape[1] == 1)
    return paths_com, label_com


def main():
    """
    Crop from SAR images 100 x 100 sub-regions centered about the lat,
    lon coordinate given by the labelled egg codes provided by
    label_egg_codes.py.
    """
    config: ConfigurationIce = get_configs()
    label_type = config.name[2]  # Todo: Get idx of label type dynamically
    ct_type = config.name[4]  # Todo: Get idx of Ct type dynamically
    path_data_batch = os.path.join(
        './Dataset/Batches',
        config.data_batch + '.json')
    path_sar_labelled = './Dataset/Labelled_SAR_Data/'
    HW = 50  # Half-width of crop.
    # 50 is chosen to make 100 x 100 crops for reasons described in MDPI paper.

    make_dirs(path_sar_labelled)

    with open(path_data_batch, 'rb') as f:
        batch_paths = json.load(f)
    sar_paths = batch_paths['SAR']
    dex_paths = batch_paths[config.dex_type]
    dex_files = [
        p.replace(
            'Raw_DEX',
            f'Labelled_DEX_Data/Label_Type_{label_type}/Ct_Type_{ct_type}'
        )
        + '_LABELLED.pickle' for p in dex_paths
    ]

    polarizations = config.pol
    if 'HH_DIV_HV' in config.pol or 'HV_DIV_HH' in config.pol:
        polarizations = ['HH', 'HV']  # Ratio is done in dataset.py
    sar_set, label_set = [], []
    for pol in polarizations:
        sar_files = []
        for i, p in enumerate(sar_paths):
            raw_sar_path = p + f'_HH_HV_SGF/imagery_{pol}.tif'
            if os.path.isfile(raw_sar_path):
                sar_files.append(raw_sar_path)
            else:
                raise FileNotFoundError(
                    f'No such file: {raw_sar_path}.\n'
                    f'Invalid polarization in configuration or '
                    f'imagery_{pol}.tif must be created.'
                )
        crop_pl = generate_crops(sar_files, dex_files, HW,
                                 config.dex_type, pol)  # Crop paths & labels
        crop_pl = crop_pl[np.argsort(crop_pl[:, 0])]
        sar_set.append(crop_pl[:, 0])
        label_set.append(crop_pl[:, 1].astype(np.float))
    if len(config.pol) == 1:
        sar_set = np.column_stack(sar_set)
        label_set = np.column_stack(label_set)
    elif len(config.pol) > 1:
        # Intersection between lists of crops returned from generate_crops()
        # for different polarizations contained in sar_set is needed since
        # for the same SAR scene, sub-regions in one polarization do not pass
        # the 80% while in another polarization it does.
        sar_set, label_set = intersection(sar_set, label_set, config.pol)

    # Dataset consisting of paths to SAR crops and their labels.
    ds_pl = {'sar_paths': sar_set, 'labels': label_set}
    ds_code = config.name.split('_')[0]
    fname = os.path.join(path_sar_labelled, ds_code[:-1] + 'X' + '.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(ds_pl, f)

    # Remove .aux.xml files created by opening the SAR image with GDAL.
    # rm_cmd = f'rm {PATH_SAR_SET}/*.aux.xml'
    # os.system(rm_cmd)


if __name__ == '__main__':
    main()
    print('Program Terminated')
