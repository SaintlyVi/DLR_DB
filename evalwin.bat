#!/bin/bash

set root=C:\Users\CKAN\Anaconda3
call %root%\Scripts\activate.bat %root%

clear
echo "Ready to roll."

python evalClusters.py exp5_kmeans_zero-one features1
python evalClusters.py exp5_kmeans_demin features1
python evalClusters.py exp4_kmeans_zero-one features1