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

### Main environments that have been tested
* python 3.7.13
* numpy 1.21.5
* scipy 1.7.3
* matplotlib 3.5.2
* torch 1.12.1
* pandas 1.3.5
* sklearn 1.0.2


### How to use
#### Overview
There are two parts for this work, model training and ensemble learning.
1. Model training: this part is implemented using python, which is mainly based on [geatpy](https://github.com/geatpy-dev/geatpy). Further, we modified it so that the code supports individuals (solutions) in a population being neural networks during optimisation process.

2. Ensemble learning: model selection strategies are implemented using matlab mainly based on [PlatEMO](https://github.com/BIMK/PlatEMO). Analysis code is also provided.
#### Set running environment

In `FairEMOL/__init__.py`, if your system is linux, please choose

````lib_path = __file__[:-11] + 'core/Linux/lib64/v3.7/'````

If your syetem is windows, please choose

````lib_path = __file__[:-11] + 'core/Windows/lib64/v3.7/'  ````

#### Load/Add dataset
1. Data pre-processing is based on the [approach](https://github.com/algofairness/fairness-comparison).
   1. Put the raw data in `FairEMOL/EvolutionaryCodes/data/raw`
   2. Write the code to process raw date in `FairEMOL/EvolutionaryCodes/data/objects`, e.g., `German.py`
   3. Run the code `FairEMOL/EvolutionaryCodes/data/objects/ProcessedData.py` 
   4. Obtain the processed data in `FairEMOL/EvolutionaryCodes/data/preprocessed`, e.g., `german_numerical-for-NN.csv`
   5. Restore processed data in the folder `FairEMOL/EvolutionaryCodes/data`, e.g., `German`
2. Use `FairEMOL/EvolutionaryCodes/load_data.py` to splite the data into training data, validation data, ensemble data (if you need) and test data. You can also set `save_csv=1` to manually store them.
3. Add the dataset's name in `FairEMOL/EvolutionaryCodes/data/objects/list.py`

#### Run algorithm
1. Set algorithmic parameters in `Fairness_main.py`
2. Run `Fairness_main.py`

#### Obtain results
* Results will be stored in the folder `Result/time_of_starting_the_run`, e.g., `2022-10-25-16-13-29`. 
  * `Parameters.txt`: record all the paremeters
  * `allobjs/ALL_Objs_{valid, test, ensemble, test}_genxxx_sofar.csv`: three or four `csv` file to record the objectives of all the individuals during optimisation process
  * `detect/`: record (1) the details of the dataset and (2) the objectives of all the individuals during optimisation process every `logMetric` (set in `Fairness_main.py`) generation
  * `nets/`: the neural network files of all the individuals during optimisation process every `logMetric` (set in `Fairness_main.py`) generation

#### Ensemble Learning
Based on the trained ML models, ensemble outputs considering some ML models can be calculated. This part is implemented using Matlab and the codes are in the folder `EnsembleCodes`.
1. `compute_info`: Process the results of each run in the folder `Result/time_of_starting_the_run/detect/` 
   1. Load data stored in the folder `Result/time_of_starting_the_run/detect/`
   2. Apply ensemble strategies and calculate the ensemble outputs
   3. Store the results of each run in the file `time_of_starting_the_run.mat`
3. `merge_info`: merge all the ensemble results over all the runs for a dataset into one `.mat` file
4. `analyse_info`: based on the results, process analysis
