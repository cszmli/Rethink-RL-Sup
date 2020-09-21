## Data

unzip [zip](https://drive.google.com/file/d/1zds1y0ZwmJsIaTeBNKDLIQeRl6SK1Fes/view?usp=sharing) under `data` directory.

There are two slightly different state trackers in this repository, tracker_new.py and tracker_old.py. The results in the paper is based on tracker_old.py. Just replace file tracker.py with the one that you want to use. 

## DiaMultiDense and DiaAdv
The params for DiaAdv can be found in file: diaadv_emnlp_reproduce. The first five lines are for DiaMultiDense while the left five lines are for DiaAdv. Other hyperparameters can be found in the file: utils.py 
```
python -u main.py $line_params
```
replace $line_params with one line from file diaadv_emnlp_reproduce.

## DiaSeq
The params for DiaSeq can be found in file: diaseq_emnlp_reproduce. Other hyperparameters can be found in the file: utils.py 
```
python -u main_seq.py $line_params
```
replace $line_params with one line from file diaseq_emnlp_reproduce.


## Requirements

python 3.6

pytorch 1.0

CPU
