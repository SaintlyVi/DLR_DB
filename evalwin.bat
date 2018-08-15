#!/bin/bash

set root=C:\Users\CKAN\Anaconda3
call %root%\Scripts\activate.bat %root%

clear
echo "Ready to roll."

python evalClusters.py exp5_kmeans_unit_norm features1
python evalClusters.py exp5_kmeans_unit_norm features2
python evalClusters.py exp5_kmeans_unit_norm features3