# FairEMOL - Mitigating Unfairness via Evolutionary Multiobjective Optimisation Learning

This is the code for the paper "Mitigating Unfairness via Evolutionary Multi-objective Ensemble Learning" in 
IEEE Transactions on Evolutionary Computation, in which we proposed a framework based on evolutionary multi-objective 
optimisation learning. The PDF file is available at the [IEEE website](https://ieeexplore.ieee.org/document/9902997).

**Personal Use Only. No Commercial Use**

Please consider citing this work if you use this repository. The bibtex is as followes:

````
@ARTICLE{9902997,
  author={Zhang, Qingquan and Liu, Jialin and Zhang, Zeqi and Wen, Junyi and Mao, Bifei and Yao, Xin},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={Mitigating Unfairness via Evolutionary Multi-objective Ensemble Learning}, 
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TEVC.2022.3209544}}
````

### Discussions about this project

### Environments that have been tested
* python 3.7
* numpy 
* scipy 
* matplotlib 
* torch
* pandas
* sklearn


### How to use

#### Load dateset
1. put the raw data in `FairEMOL/Fair_add/data/raw`
2. write the code to process raw date in `FairEMOL/Fair_add/data/objects`, e.g., `german.py`
3. obtain the processed data in `FairEMOL/Fair_add/data/objects`, e.g., `german_numerical-for-NN.csv`
4. restore processed data in the folder `FairEMOL/Fair_add/data`, e.g., `German`
5. use `FairEMOL/Fair_add/load_data.py` to splite the data into training data, validation data, ensemble data (if you need) and test date . You can also set`save_csv=1` to store them.
6. add the data's name in `FairEMOL/Fair_add/data/objects/list.py`

#### Set parameters
1. set algorithmic parameters in `Fairness_main.py`
2. run `Fairness_main.py`

#### Obtain results
* results will be stored in the folder `Result/time_of_starting_the_run` 
  * `Parameters.txt`: record all the paremeters
  * `allobjs/ALL_Objs_{valid, test, ensemble, test}_genxxx_sofar.csv`: three or four `csv` file to record the objectives of all the individuals during optimising process
  * `detect`: record the details of the dataset and the objectives of all the individuals during optimising process every `logMetric` in `Fairness_main.py` generation
  * `nets`: the neural network files of all the individuals during optimising process every `logMetric` in `Fairness_main.py` generation