# HGTUL

## Requirements
* Python >= 3.7 (Tested on 3.7.16)
* Pytorch >= 1.7.1 (Tested on 1.7.1)
* torch-geometric == 2.0.4
* scikit-learn == 1.0.2
## File Descriptions
* raw_data : Contains the dataset required for model training, where the folder named 'Dataset-Number' refers to the dataset after data balancing, and 'Dataset-Number-o' refers to the original dataset.
* main.py : training the **HGTUL** model.
* dataset.py : process data for model training.
* model.py : defining the classes of the models.
* utils.py : containing lots of tool functions.

## How to reproduce the results in the paper?

To reproduce the results on the NYC(users=500) dataset, just run with the following command.
```shell script
python main.py --dataset NYC --user_number 500 --num_epochs 50 --deviceID 0
```
To reproduce the results on the NYC(users=1000) dataset, just run with the following command.
```shell script
python main.py --dataset NYC --user_number 1000 --num_epochs 50 --deviceID 0
```
To reproduce the results on the JKT(users=500) dataset, just run with the following command.
```shell script
python main.py --dataset JKT --user_number 500 --num_epochs 50 --deviceID 0
```
To reproduce the results on the JKT(users=1000) dataset, just run with the following command.
```shell script
python main.py --dataset JKT --user_number 1000 --num_epochs 50 --deviceID 0
```
To reproduce the results on the GOWALLA(users=500) dataset, just run with the following command.
```shell script
python main.py --dataset GOWALLA --user_number 500 --num_epochs 50 --deviceID 0
```
To reproduce the results on the GOWALLA(users=1000) dataset, just run with the following command.
```shell script
python main.py --dataset GOWALLA --user_number 1000 --num_epochs 50 --deviceID 0
```

