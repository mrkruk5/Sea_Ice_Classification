import os
import sys
import argparse
import json
import traceback


def get_configs():
    desc = 'Configuration arguments provided at run time from the CLI'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-c',
        metavar='',
        type=str,
        default=None,
        help='Example: python3 script.py -c config_file.json'
    )
    args = parser.parse_args()
    if args.c:
        config_name = args.c.split('/')[-1].split('.')[0]
        # noinspection PyBroadException
        try:
            with open(args.c, 'r') as f:
                config_dict = json.load(f)
                config_dict['config_path'] = args.c
                config_dict['name'] = config_name
                config = ConfigurationIce(config_dict)
        except json.decoder.JSONDecodeError as e:
            print(f'Error occurred loading the provided file with JSON:\n'
                  f'File: {args.c}\n'
                  f'Error: {e}')
            sys.exit(1)
        except Exception:
            traceback.print_exc()
        else:  # If no errors
            return config
    else:
        print('No configuration file provided.')
        parser.print_help()
        sys.exit(1)


class Configuration(object):
    def __init__(self, config: dict):
        """
        The Configuration class contains settings of variables used to
        control the program flow.
        Args:
            config: A dictionary containing the configuration variables and
                their setting.
        Attrs:
            config_path: A string of the path to the configuration file.
            name: A string of the name of the configuration file.
            dataset_config: A sub-dictionary of config that contains
                configuration variables pertaining to the dataset.
            model_config: A sub-dictionary of config that contains
                configuration variables pertaining to the model.
            data_batch: A string denoting a JSON file that contains the data
                file paths to be used to make a dataset.
            model: A string of the model to be used for training.
            loss: A string of the loss function to be used to train the model.
            metric: A list of strings containing the metrics to be used to
                assess the model.
            batch_size: An int indicating the number of samples to be used in a
                training batch.
            epochs: An int indicating the number of times the dataset is
                passed over for training.
            early_stop_patience: An int indicating the number of epochs to
                let pass without improvement in the validation loss before
                stopping.
            reduce_lr_patience: An int indicating the number of epochs to
                let pass without improvement in the validation before reducing
                the learning rate.
        """
        self.config_path = config['config_path']
        self.name = config['name']
        self.dataset_config = config['dataset_config']
        self.model_config = config['model_config']

        # Unpack dataset_config
        self.data_batch = self.dataset_config['data_batch']

        # Unpack model_config
        self.model = self.model_config['model']
        self.loss = self.model_config['loss']
        self.metric = self.model_config['metric']
        self.batch_size = self.model_config['batch_size']
        self.epochs = self.model_config['epochs']
        self.es_patience = self.model_config['early_stopping_patience']
        self.lr_patience = self.model_config['reduce_lr_patience']


class ConfigurationIce(Configuration):
    def __init__(self, config: dict):
        """
        The ConfigurationIce class contains settings of variables
        specific to the Sea Ice classification project used to control the
        program flow.
        Args:
            config: A dictionary containing the configuration variables and
                their setting.
        Attrs:

        """
        super().__init__(config)
        self.dex_type = self.dataset_config['dex_type']
        self.code_to_label = self.dataset_config['code_to_label']
        self.label_names = self.dataset_config['label_names']
        self.pol = self.dataset_config['pol']
        self.Ct = self.dataset_config['Ct']
        self.pre_processing = self.dataset_config['pre_processing']


class Logger(object):
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')
        self.is_open = True

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message.replace('\b', ''))

    def flush(self):
        pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal
        self.is_open = False


def make_dirs(path: str):
    result = False
    # noinspection PyBroadException
    try:
        os.makedirs(path)
        result = True
    except FileExistsError:
        result = False
    except Exception:
        traceback.print_exc()
    finally:
        return result
