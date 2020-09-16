# To do
* Check how the measure is made for each network, stations, instruments, variables.

# Folder description

This folder contains the scripts and library for weather station quality control

# Procedure of the quality control
1. Clean the files of each stations with basic consistency check and variable limits
2. Get stations data in selected domain and return a dataframe for each variables
3. Visual selection and QA analysis assurance check
4. Flag megaconsistency stations (e.g. all event == 0)

