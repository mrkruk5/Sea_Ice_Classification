import os
from glob import glob
import json
from utils import make_dirs


if __name__ == '__main__':
    PATH_SAR = './Dataset/Raw_SAR/'
    PATH_DEX = './Dataset/Raw_DEX/'
    PATH_BATCH = './Dataset/Batches/'
    make_dirs(PATH_BATCH)

    sar_list = sorted(glob(os.path.join(PATH_SAR, 'RS2*')))
    sar_list = [p.replace('_HH_HV_SGF', '') for p in sar_list]
    dexa_list = sorted(glob(os.path.join(PATH_DEX, 'DEXA', '*.dex')))
    dexa_list = [p.replace('.dex', '') for p in dexa_list]
    dexi_list = sorted(glob(os.path.join(PATH_DEX, 'DEXI', '*.dex')))
    dexi_list = [p.replace('.dex', '') for p in dexi_list]
    batch = {'SAR': sar_list, 'DEXA': dexa_list, 'DEXI': dexi_list}

    with open(os.path.join(PATH_BATCH, 'Batch_4.json'), 'wt') as f:
        json.dump(batch, f, indent=4)

    print('Program Terminated')
