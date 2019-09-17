from core_functions.auto_kernel_learning import askl
import core_functions.optimal_parameters as optimal_parameters
import os
import numpy as np
import pickle
import math

datasets = ['mnist']
n_repeats = 1

results = [[], [], [], [], []]
for dataset in datasets:
    parameter_dic = optimal_parameters.get_parameter(dataset)
    for t in range(n_repeats):
        exp1_result = {}
        # # 1. SK
        parameter_dic_SK = parameter_dic.copy()
        parameter_dic_SK['stationary'] = True
        parameter_dic_SK['back_propagation'] = False
        parameter_dic_SK['regular_type'] = 'fro'
        parameter_dic_SK['learning_rate'] = 1e-2
        exp1_result['SK'] = askl(parameter_dic_SK)
        results[0].append(exp1_result['SK']['test_accuracy'])

        # 2. NSK
        parameter_dic_NSK = parameter_dic.copy()
        parameter_dic_NSK['stationary'] = False
        parameter_dic_NSK['back_propagation'] = False
        parameter_dic_NSK['regular_type'] = 'fro'
        parameter_dic_NSK['learning_rate'] = 1e-2
        exp1_result['NSK'] = askl(parameter_dic_NSK)
        results[1].append(exp1_result['NSK']['test_accuracy'])

        # 3. SKL
        parameter_dic_SKL = parameter_dic.copy()
        parameter_dic_SKL['stationary'] = True
        parameter_dic_SKL['back_propagation'] = True
        parameter_dic_SKL['regular_type'] = 'fro'
        exp1_result['SKL'] = askl(parameter_dic_SKL)
        results[2].append(exp1_result['SKL']['test_accuracy'])

        # 4. NSKL
        parameter_dic_NSKL = parameter_dic.copy()
        parameter_dic_NSKL['stationary'] = False
        parameter_dic_NSKL['back_propagation'] = True
        parameter_dic_NSKL['regular_type'] = 'fro'
        exp1_result['NSKL'] = askl(parameter_dic_NSKL)
        results[3].append(exp1_result['NSKL']['test_accuracy'])

        # 5. ASKL
        parameter_dic_ASKL = parameter_dic.copy()
        exp1_result['ASKL'] = askl(parameter_dic_ASKL)
        results[4].append(exp1_result['ASKL']['test_accuracy'])

        with open('./results/exp1/' + dataset + '_' + str(t), 'wb') as handle:
                pickle.dump(exp1_result, handle)

print(results)
np_results = np.array(results)
print('&%.2f$\pm$%.2f   &%.2f$\pm$%.2f  &%.2f$\pm$%.2f  &%.2f$\pm$%.2f  &%.2f$\pm$%.2f' % (-np_results.mean(1)[0], np_results.std(1)[0], -np_results.mean(1)[1], np_results.std(1)[1], -np_results.mean(1)[2], np_results.std(1)[2], -np_results.mean(1)[3], np_results.std(1)[3], -np_results.mean(1)[4], np_results.std(1)[4]))