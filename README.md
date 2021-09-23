# Sea_Ice_Classification

For a quick start, type "./run.sh" within the Sea_Ice_Classification working
directory and this will run the necessary code to run an experiment from start
to finish.
This will run DEX_reader.py, label_egg_codes.py, SAR_crop_per_DEX.py and
model.py sequentially.

--------------------------------------------------------------------------------

DEX_reader.py:
Parses files with the ".dex" extension located in ./Dataset/Raw_DEX which
contain Canadian Ice Service ice chart data. The data in these files are
egg codes with an associated lat/lon coordinate. DEX_reader.py outputs a pickle
file containing an ndarray of lat/lon coordinates with labels to egg codes
along with a dict of the mapping between egg codes and their label. A pickle
file is produced for each DEX file and these are stored in
./Dataset/Parsed_DEX_Data (not uploaded, but created during execution).

label_egg_codes.py:
Re-labels the egg codes output from DEX_reader.py according to the provided
configuration file and stage of development feature of the egg. A pickle file of
an ndarray of the remapped labels along with their lat/lon coords are stored in
./Dataset/Labelled_DEX_Data (not uploaded, but created during execution).

SAR_crop_per_DEX.py:
Crops from SAR images 100 x 100 sub-regions centered about the lat/lon coord
given by the labelled egg codes provided by label_egg_codes.py. The cropped
sub-regions are stored in ./Dataset/Cropped_SAR_Data. The crops to be used
for the dataset described by the configuration file are summarized in an ndarray
of paths along with a 1-to-1 map to another ndarray containing the labels for
each crop. These ndarrays are stored in ./Dataset/Labelled_SAR_Data. Both
directories have not been uploaded for storage purposes, but are created during
execution.

model.py:
Calls dataset.py, which gathers the crops and labels outlined by the output of
SAR_crop_per_DEX.py into large ndarrays contained in a Dataset object. The
Dataset object is stored in ./Dataset/Packaged_Dataset (also not uploaded, but
generated during execution) for future use. A model is then trained using the
Dataset object and results are stored in ./Results (not uploaded) and named
after the configuration file.

--------------------------------------------------------------------------------

Auxiliary files:
analysis_tools.py: Used to get result statistics and figures.
calibration.py: Used for calibration of SAR images during pre-processing. Also
    contains a function to create Ground Control Points (GCPs) to properly
    render crops in a GIS application.
make_batch_paths.py: Used to create the Batch_X.json files, which define the
    source files to use for a particular experiment.
metrics.py: Used to supply metrics to the model. Custom metrics can be written
    here.
networks.py: Network definitions are stored here. Add new networks here.
product_info.py: Contains getter functions that read SAR product files for
    specific auxiliary data.
utils.py: Contains a variety of utility functions and classes.

--------------------------------------------------------------------------------

Other important info:
Definition of the naming convention used to identify the experiments conducted
can be found in ./Configs/config_naming_scheme.json.

The packages used in the conda environment used for this project can be found
in ./python_env_sea_ice.yml, which was created with the following command (with
the appropriate conda environment activated):
conda env export > python_env_sea_ice.yml
This environment may be re-loaded with:
conda env create -f python_env_sea_ice.yml
However, despite the fact that conda is meant to be portable, this is platform
dependent and may not work on another workstation.
