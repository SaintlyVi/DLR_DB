# SA Domestic Load Research Data Access & Analysis

This repository is a submodule of the [DLR Intelligent System](https://github.com/SaintlyVi/DLR_DB). It contains modules and functions for data extraction, manipulation and analysis of the DLR MSSQL database.

To use this repository as a standalone project:

1. Make a parent direcotry and call it DLR_DB.
2. Clone this repo from git: `git clone https://github.com/ERC-data/DLR_DB.src.git src`

### Data Handling

To run the project without a MSSQL installation of the DLR database, access to a protected data repository is required. You can request access to this repo from [@saintlyvi](wiebke.toussaint@uct.ac.za).

Feather has been chosen as the format for temporary data storage as it is a fast and efficient file format for storing and retrieving data frames. It is compatible with both R and python. Feather files should be stored for working purposes only as the file format is not suitable for archiving. All feather files have been built under `feather.__version__ = 0.4.0`. If your feather package is of a later version, you may have trouble reading the files and will need to reconstruct them from the raw MSSQL database. Learn more about [feather](https://github.com/wesm/feather).

### Data Retrieval and Processing

If you have access to an MSSQL installation of the DLR database, you can access the raw survey and profile data.

1. Clone the reposistory as described above.
2. Use `saveTables()` in fetch_data.py to save tables as feather files * 
3. Use `saveProfiles()` in fetch_data.py to save load profiles as feather files *
4. Use `reduceRawProfiles()` and `saveHourlyProfiles()` in tswrangler.py to reduce and save 5min data to hourly timeseries

### Data Exploration

Once the data has been extracted from the database and saved in the appropriate directories, you can explore the data.
