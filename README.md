# SA Domestic Load Research Intelligent System
This is the root directory for the South African Domestic Load Research Intelligent System. It contains the following submodules, which will be empty directories when cloning this repository:
1. src
2. libpgm

To fetch the data from the submodules you must run `git submodule init` and `git submodule update` or alternatively pass `--recursive` to the `git clone` command. Learn more about [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

## Interacting with the DLR Data

### Data Handling
To run the scripts without a MSSQL database installation of the DLR project, access to a protected data repository is required. It must be included as the `\data` subdirectory to this main repository. Access to the data repo can be requested from [@saintlyvi](wiebke.toussaint@uct.ac.za).

### Navigating the System
The DLR NOTEBOOK files are jupyter notebooks that provide a step-by-step guide for processing, analysing and visualising the domestic load data. 

Learn more about installing and running [jupyter notebooks](http://jupyter.readthedocs.io/en/latest/install.html).