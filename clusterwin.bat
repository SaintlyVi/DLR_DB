#!/bin/bash

set root=C:\Users\CKAN\Anaconda3
call %root%\Scripts\activate.bat %root%

clear
echo "Ready to roll."

#python runClusters.py exp3_kmeans -top 5
python runClusters.py exp3_som -top 5
python runClusters.py exp3_som+kmeans -top 5