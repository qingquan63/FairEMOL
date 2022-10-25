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
The code implementation is based on [geatpy](https://github.com/geatpy-dev/geatpy). Further, we modified it so that the code supports individuals in a population being neural networks.

### Environments that have been tested
* python 3.7
* numpy 
* scipy 
* matplotlib 
* torch
* pandas
* sklearn


### How to use

#### Set running environment

In `FairEMOL/__init__.py`, if your system is linux, please choose

````lib_path = __file__[:-11] + 'core/Linux/lib64/v3.7/'````

If your syetem is windows, please choose

````lib_path = __file__[:-11] + 'core/Windows/lib64/v3.7/'  ````

#### Load dateset
1. Put the raw data in `FairEMOL/EvolutionaryCodes/data/raw`
2. Write the code to process raw date in `FairEMOL/EvolutionaryCodes/data/objects`, e.g., `German.py`
3. Obtain the processed data in `FairEMOL/EvolutionaryCodes/data/preprocessed`, e.g., `german_numerical-for-NN.csv`
4. Restore processed data in the folder `FairEMOL/EvolutionaryCodes/data`, e.g., `German`
5. Use `FairEMOL/EvolutionaryCodes/load_data.py` to splite the data into training data, validation data, ensemble data (if you need) and test data. You can also set`save_csv=1` to manually store them.
6. Add the dataset's name in `FairEMOL/EvolutionaryCodes/data/objects/list.py`

#### Run algorithm
1. Set algorithmic parameters in `Fairness_main.py`
2. Run `Fairness_main.py`

#### Obtain results
* Results will be stored in the folder `Result/time_of_starting_the_run`, e.g., `2022-10-25-16-13-29`. 
  * `Parameters.txt`: record all the paremeters
  * `allobjs/ALL_Objs_{valid, test, ensemble, test}_genxxx_sofar.csv`: three or four `csv` file to record the objectives of all the individuals during optimisation process
  * `detect/`: record (1) the details of the dataset and (2) the objectives of all the individuals during optimisation process every `logMetric` in `Fairness_main.py` generation
  * `nets/`: the neural network files of all the individuals during optimisation process every `logMetric` in `Fairness_main.py` generation