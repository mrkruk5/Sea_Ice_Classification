import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import get_configs, ConfigurationIce, make_dirs


def map_eggs_to_labels(eggs, c: ConfigurationIce):
    """
    Creates a dictionary that maps the arbitrary values in eggs that were set
    in DEX_reader.py to appropriate labels per the stage of development
    feature in the egg.
    :param eggs: Dict of egg codes (stored as a tuple in dict keys) and their
        enumerated values by order of appearance.
    :param c: ConfigurationIce object.
    :return: Dict that maps values in eggs to labels
    """
    d = dict(zip(eggs.values(), eggs.values()))
    for key, val in eggs.items():
        if type(key) is tuple:
            egg = dict(key)
            ct = convert_code(egg['Ct'])
            if ct >= c.Ct:
                if 'Sa' in egg and 'Sb' in egg and 'Sc' in egg:
                    sod = [egg['Sa'], egg['Sb'], egg['Sc']]  # Stage of dev.
                elif 'Sa' in egg and 'Sb' in egg and 'Sc' not in egg:
                    sod = [egg['Sa'], egg['Sb']]  # Stage of dev.
                elif 'Sa' in egg and 'Sb' not in egg and 'Sc' not in egg:
                    sod = [egg['Sa']]  # Stage of dev.
                else:
                    raise RuntimeWarning(f'Unusual egg: {egg}.')
                labels = np.array([label_code(s, c.code_to_label) for s in sod])
                same = np.all(labels == labels[0])
                if same:
                    d[val] = labels[0]
                else:
                    d[val] = None
            else:
                d[val] = None
        else:
            # For keys == POINT NOT COVERED BY POLYGON, LAND, OW, BW, IF,
            #             ICEGLACE, and FAST.
            d[val] = c.code_to_label[key]
    return d


def label_code(code: str, d: dict) -> int:
    code = convert_code(code)
    return d[str(code)]


def convert_code(code: str) -> int:
    try:
        code = int(code)
    except ValueError as e:
        # Below are arbitrary operations to distinguish codes with characters
        # that cannot be cast to int.
        if '.' in code:
            code = int(code[:-1]) * 10
        elif '+' in code:
            code = int(code[:-1]) + 0.5
        else:
            print(e)
    finally:
        return code


def main():
    """
    Label data saved by DEX_reader.py per config.
    """
    config = get_configs()
    label_type = config.name[2]  # Todo: Get idx of label type dynamically
    ct_type = config.name[4]  # Todo: Get idx of Ct type dynamically
    path_data_batch = os.path.join(
        './Dataset/Batches',
        config.data_batch + '.json')
    path_labelled = os.path.join(
        './Dataset/Labelled_DEX_Data',
        'Label_Type_' + label_type,
        'Ct_Type_' + ct_type,
        config.dex_type)

    make_dirs(path_labelled)

    with open(path_data_batch, 'rb') as f:
        batch_paths = json.load(f)
    paths = batch_paths[config.dex_type]
    # Parsed_DEX_Data contains output of DEX_reader.py
    files = [p.replace('Raw_DEX', 'Parsed_DEX_Data') + '.pickle' for p in paths]

    for file in files:
        new_name = file.split('/')[-1].split('.')[0] + '_LABELLED.pickle'
        fname = os.path.join(path_labelled, new_name)
        if os.path.isfile(fname):
            print(
                f'Labelled type {label_type} and Ct type {ct_type} egg codes '
                f'given in config {config.name} already exist for '
                f'{file.split("/")[-1]}.')
            with open(fname, 'rb') as f:
                data_labelled = pickle.load(f)
            if data_labelled is not None:
                labels = np.unique(data_labelled[:, 2])
                # Wanted to find img with all 3 classes for MDPI paper
                if labels.size == 3:
                    print(fname)

        else:
            print(f'Labelling file {file}')
            timestamp = file.split('/')[-1].split('_')[0]
            title = 'DEX ' + 'Y' + timestamp[:4] + ' M' + timestamp[4:6] + \
                    ' D' + timestamp[6:8] + ' h' + timestamp[8:10] + \
                    ' m' + timestamp[10:12] + ' s' + timestamp[12:14]
            with open(file, 'rb') as f:
                dex_data = pickle.load(f)
            data = dex_data['data']

            data_labelled = np.copy(data)
            egg_codes = dex_data['egg_codes']
            mapping = map_eggs_to_labels(egg_codes, config)
            # Update data_labelled with appropriate labels according to mapping
            for label in set(mapping.values()):
                # Get all keys that map to the same value.
                test_conditions = [key for key, val in mapping.items()
                                   if val == label]
                # Test each element of the 3rd column in data to the values in
                # test_conditions and replace them with the mapping value.
                data_labelled[np.in1d(data[:, 2], test_conditions), 2] = label
            # Remove all rows that contain NaN (values set to None in mapping).
            data_labelled = data_labelled[~np.isnan(data_labelled).any(axis=1)]
            if data_labelled.size == 0:
                print(f'File {file} does not contain data that fall under the '
                      f'label and concentration categories specified by '
                      f'{config.name}.')
                data_labelled = None
            else:
                # Visualize simplified label space.
                fig, ax = plt.subplots()
                pt_size = 0.05
                if config.dex_type == 'DEXI':
                    pt_size = 0.4
                scatter = ax.scatter(data_labelled[:, 1], data_labelled[:, 0],
                                     c=data_labelled[:, 2],
                                     s=pt_size)
                legend = ax.legend(*scatter.legend_elements(),
                                   title='Egg Code Labels',
                                   loc='lower right',
                                   frameon=False)
                ax.add_artist(legend)
                ax.set_title(title)
                ax.set_ylabel('Latitude')
                ax.set_ylim(bottom=np.min(data[:, 0]), top=np.max(data[:, 0]))
                ax.set_xlabel('Longitude')
                ax.set_xlim(left=np.min(data[:, 1]), right=np.max(data[:, 1]))
                plt.savefig(fname.replace('.pickle', '.png'),
                            bbox_inches='tight')
                # plt.show()
                plt.close()

            # Save data.
            with open(fname, 'wb') as f:
                pickle.dump(data_labelled, f)


if __name__ == '__main__':
    main()
    print('Program Terminated')
