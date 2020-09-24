# Rethinking Supervised Learning and Reinforcement Learning in Task-Oriented Dialogue Systems
This is the codebase for paper: "[Rethinking Supervised Learning and Reinforcement Learning in Task-Oriented Dialogue Systems](https://arxiv.org/abs/2009.09781)".


## Requirements

python 3.6

pytorch 1.0

CPU


## Data

unzip [file](https://drive.google.com/file/d/1HhVmWnwsm651n1sBUvGsx-PJ1yB-HhRG/view?usp=sharing) under `data` directory.

There are two slightly different state trackers in this repository, tracker_new.py and tracker_old.py. The result in the paper is based on tracker_old.py. Just replace file tracker.py with the one that you want to use. 

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


If you use the code for dialogue policy learning, feel free to cite our publication [Rethinking Supervised Learning and Reinforcement Learning in Task-Oriented Dialogue Systems](https://arxiv.org/abs/2009.09781):
``` 
@article{li2020rethink,
  title={Rethinking Supervised Learning and Reinforcement Learning in Task-Oriented Dialogue Systems},
  author={Li, Ziming and Kiseleva, Julia and de Rijke, Maarten},
  journal={Findings of EMNLP 2020},
  year={2020}
}

```


This code is based on the source code of Ryuichi's [GDPL work](https://github.com/truthless11/GDPL).


