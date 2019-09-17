from core_functions.auto_kernel_learning import askl
import core_functions.optimal_parameters as optimal_parameters
import os
import pickle
import math

def generate_path(parameter_dic):
    return 'results/' + '_'.join([parameter_dic['dataset'], 
            str(parameter_dic['std1']), 
            str(parameter_dic['std2']), 
            str(parameter_dic['lambda_A']), 
            str(parameter_dic['lambda_B']), 
            str(parameter_dic['D']), 
            str('stationary' if parameter_dic['stationary'] else 'nonstationary'),
            str('bp' if parameter_dic['back_propagation'] else 'assigned'), 
            str(parameter_dic['batch_size']), 
            str(parameter_dic['T']), 
            str(parameter_dic['record_batch'])])

parameter_dic = optimal_parameters.get_parameter('cadata')

optimal_result ={'test_accuracy' : -1e10}
## 1. Tune kernel parameters
std1_set = [math.pow(2, t) for t in range(-9, 0)]
for std1 in std1_set:
    parameter_dic['std1'] = std1
    parameter_dic['std2'] = std1
    result_dict = askl(parameter_dic.copy())
    if optimal_result['test_accuracy'] < result_dict['test_accuracy']:
        optimal_result = result_dict
print(optimal_result)
parameter_dic = optimal_result['parameter_dic'].copy()

# 2. Tune lambda
lambda_A_set = [math.pow(10, t) for t in range(-5, -2)]
lambda_B_set = [math.pow(10, t) for t in range(-5, -2)]
for lambda_A in lambda_A_set:
    for lambda_B in lambda_B_set:
        parameter_dic['lambda_A'] = lambda_A
        parameter_dic['lambda_B'] = lambda_B
        result_dict = askl(parameter_dic.copy())
        if optimal_result['test_accuracy'] < result_dict['test_accuracy']:
            optimal_result = result_dict
print(optimal_result)
save_path = generate_path(optimal_result['parameter_dic'])
with open(save_path, 'wb') as handle:
        pickle.dump(optimal_result, handle)