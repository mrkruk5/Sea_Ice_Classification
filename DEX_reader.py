import os
import json
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from utils import get_configs, make_dirs


def sort_lat_lon(element):
    return element[0], element[1]


def read_dex(file):
    data = []
    all_egg_codes = {
        'FAST': -7, 'ICEGLACE': -6, 'BW': -5, 'OW': -4, 'IF': -3, 'LAND': -2,
        'POINT NOT COVERED BY POLYGON': -1,
    }
    format_line = '(1X,F10.5,1X,F10.5,1X,2A2,24(1X,1A2))'  # found in DEX file
    formatting = re.search(r'\((.*)\)', format_line).group(1)
    code_map = {
        1: 'Y', 2: 'X',
        3: 'Ct', 4: 'Ca', 5: 'Sa', 6: 'Fa',
        7: 'Cb', 8: 'Sb', 9: 'Fb',
        10: 'Cc', 11: 'Sc', 12: 'Fc',
        13: 'So', 14: 'Sd', 15: 'STRIPS',
        16: 'Ct1', 17: 'Ca1', 18: 'Sa1', 19: 'Fa1',
        20: 'Cb1', 21: 'Sb1', 22: 'Fb1',
        23: 'Cc1', 24: 'Sc1', 25: 'Fc1',
        26: 'So1', 27: 'Sd1'
    }

    with open(file, 'r') as f:
        lines = f.read().splitlines()

    n_lines_w_tilde = 0
    for i, line in enumerate(lines):
        if (i + 1) == 3:  # Header lines in file
            assert line == format_line
        elif 6 <= (i + 1) <= 32:  # Header lines in file
            assert line.split()[0] == code_map[(i + 1) - 5].upper()
        elif (i + 1) >= 35 and line[0] != '$':  # Data lines in file
            if '~' in line:
                print(f'Line {i+1:5}: {line}')
                n_lines_w_tilde += 1
                # Tilde messes up line formatting, removing it fixes it.
                line = line.replace('~', '')
            lat, lon, egg_code = parse_line(line, i + 1, formatting, code_map)
            if egg_code not in all_egg_codes:
                # Add new egg as key with value incrementally increasing from
                # 0 as new eggs are added in order of appearance.
                all_egg_codes[egg_code] = len(all_egg_codes) - 7
            data.append([lat, lon, all_egg_codes[egg_code]])
        else:
            continue
    if n_lines_w_tilde > 0:
        print(f'There were {n_lines_w_tilde} lines with "~"s that '
              f'were removed to fit formatting.')
    data.sort(key=sort_lat_lon)
    data = np.array(data)
    return data, all_egg_codes


def parse_line(line, line_num, formatting, code_map):
    expected_terms = ['POINT', 'LAND', 'IF', 'OW', 'BW', 'ICEGLACE', 'FAST']
    tokens = line.split()
    egg_tokens = tokens[2:]
    try:
        lat = float(tokens[0])
        # All .dex files are assumed to be in the West hemisphere.
        lon = -float(tokens[1])
    except ValueError as e:  # If there are exceptions, run this block.
        print(f'DEX line {line_num}: ValueError: {e}')
    else:
        if egg_tokens[0] in expected_terms:
            return lat, lon, ' '.join(egg_tokens)
        else:
            chars = list(line)
            pos = 0
            code_id = 0
            egg_code = {}
            for sep in formatting.split(',', 6):
                if sep == '1X':  # 1 space character (char)
                    pos += 1
                elif sep == 'F10.5':  # 10 chars float with 5 decimals
                    # 10 chars for float may have leading whitespace
                    code = ''.join(chars[pos:pos + 10]).strip()
                    code_id += 1
                    egg_code[code_map[code_id]] = code
                    pos += 10
                elif sep == '2A2':  # 2 repetitions of 2 alpha-numeric chars
                    code = ''.join(chars[pos:pos + 2]).strip()
                    code_id += 1
                    egg_code[code_map[code_id]] = code
                    pos += 2  # 1 A2
                    pos += 2  # 2 A2
                elif sep == '24(1X,1A2)':
                    # 24 reps of 1 space and 2 alpha-numeric chars
                    for _ in range(24):
                        code = ''.join(chars[pos:pos + 3]).strip()
                        code_id += 1
                        egg_code[code_map[code_id]] = code
                        pos += 3  # 1X + 1A2
                else:
                    raise ValueError(f'Unexpected separator: {sep}')
            egg_code = {k: v for k, v in egg_code.items() if v}  # Remove empty
            assert lat == float(egg_code['Y'])
            assert lon == -float(egg_code['X'])
            del egg_code['Y']
            del egg_code['X']
            egg_code = tuple(egg_code.items())  # Convert to tuple to use as key
            return lat, lon, egg_code


def main():
    config = get_configs()
    path_data_batch = os.path.join(
        './Dataset/Batches',
        config.data_batch + '.json')
    path_parsed_dex = os.path.join('./Dataset/Parsed_DEX_Data',
                                   config.dex_type)

    make_dirs(path_parsed_dex)

    with open(path_data_batch, 'rb') as f:
        batch_paths = json.load(f)
    paths = batch_paths[config.dex_type]
    files = [p + '.dex' for p in paths]

    for file in files:
        new_name = file.split('/')[-1].split('.')[0] + '.pickle'
        fname = os.path.join(path_parsed_dex, new_name)

        timestamp = file.split('/')[-1].split('_')[0]
        title = 'DEX ' + 'Y' + timestamp[:4] + ' M' + timestamp[4:6] + \
                ' D' + timestamp[6:8] + ' h' + timestamp[8:10] + \
                ' m' + timestamp[10:12] + ' s' + timestamp[12:14]
        print(f'Parsing {file}:')
        data, egg_codes = read_dex(file)
        print()

        # Visualize egg codes.
        fig, ax = plt.subplots()
        pt_size = 0.05
        if config.dex_type == 'DEXI':
            pt_size = 0.4
        scatter = ax.scatter(data[:, 1], data[:, 0], c=data[:, 2], s=pt_size)
        # Consider using a colorbar instead of a legend as in plot_img() in
        # analysis_tools.py
        legend = ax.legend(*scatter.legend_elements(num=len(set(data[:, 2]))),
                           title='Egg Code Labels',
                           title_fontsize='small',
                           fontsize='xx-small',
                           ncol=5,
                           loc='lower right',
                           frameon=False)
        ax.add_artist(legend)
        ax.set_title(title)
        ax.set_ylabel('Latitude')
        ax.set_ylim(bottom=np.min(data[:, 0]), top=np.max(data[:, 0]))
        ax.set_xlabel('Longitude')
        ax.set_xlim(left=np.min(data[:, 1]), right=np.max(data[:, 1]))
        plt.savefig(fname.replace('.pickle', '.png'), bbox_inches='tight')
        # plt.show()
        plt.close()

        # Save data.
        dataset = {'data': data, 'egg_codes': egg_codes}
        with open(fname, 'wb') as f:
            pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
    print('Program Terminated')
