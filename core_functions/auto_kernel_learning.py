# -*- coding: utf-8 -*-
import os
import torch
import torchvision
import math
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from core_functions.utils import test_in_batch, load_data, initial_weights, feature_mapping, empirical_loss, matplotlib_imshow

#device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

def initial_weights(d, D, K, bp = True, std1 = 0.05, std2 = 0.05, stationary=False, device = 'cuda:0', dtype = torch.float):
    m = Uniform(torch.tensor([0.0], device=device), torch.tensor([2*math.pi], device=device))
    if stationary:
        b1 = m.sample((1, D)).view(-1, D).to(device)
        Omega1 = torch.empty(d, D, device = device, requires_grad=bp)
        torch.nn.init.normal_(Omega1, 0, std1)
        W = torch.randn(D, K, device=device, dtype=dtype, requires_grad=True)
        return [Omega1, b1], W
    else:
        Omega1 = torch.empty(d, D, device = device, requires_grad=bp)
        Omega2 = torch.empty(d, D, device = device, requires_grad=bp)
        b1 = m.sample((1, D)).view(-1, D).to(device)
        b2 = m.sample((1, D)).view(-1, D).to(device)
        torch.nn.init.normal_(Omega1, 0, std1)
        torch.nn.init.normal_(Omega2, 0, std2)
        W = torch.randn(D, K, device=device, dtype=dtype, requires_grad=True)
        return [Omega1, Omega2, b1, b2], W

def feature_mapping(X, parameters, stationary):    
    if stationary :
        Omega1, b1 = parameters
        phi = torch.cos(X.mm(Omega1) + b1.repeat(X.shape[0], 1))*math.sqrt(2/Omega1.shape[1])
    else:
        Omega1, Omega2, b1, b2 = parameters
        phi = (torch.cos(X.mm(Omega1) + b1.repeat(X.shape[0], 1)) + torch.cos(X.mm(Omega2) + b2.repeat(X.shape[0], 1)))/math.sqrt(2*Omega1.shape[1])
    return phi

def test_in_batch(testloader, spectral_measure, stationary, W, loss_type, lambda_A, lambda_B, regular_type, device="cuda:0"):
    total = 0
    loss = 0.0
    regularaztion_W = 0.0
    regularaztion_phi = 0.0
    correct = 0
    with torch.no_grad():
        for i_batch, test_batch in enumerate(testloader):
            X_test, y_test = test_batch
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            phi = feature_mapping(X_test, spectral_measure, stationary)
            outputs = phi.mm(W)
            if regular_type == 'fro':
                regularaztion_W = lambda_A * torch.norm(W, 'fro')
                regularaztion_phi = torch.tensor(0.0)
            else:
                with torch.no_grad():
                    regularaztion_W = lambda_A * sum(torch.svd(W)[1]) #torch.norm(W, regular_type)
                regularaztion_phi = lambda_B * torch.norm(phi, 'fro')
            loss += empirical_loss(outputs, y_test, loss_type)

            if regular_type == 'fro':
                regularaztion_W += lambda_A * torch.norm(W, 'fro')
            else:
                regularaztion_W += lambda_A * sum(torch.svd(W)[1])#torch.norm(W, regular_type)
                regularaztion_phi += lambda_B * torch.norm(phi, 'fro')

            
            if W.shape[1] > 1 :
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y_test).sum().item()*100
                total += y_test.size(0)
            else :
                correct += math.sqrt(empirical_loss(outputs, y_test, 'mse').item())
                total += 1

        return  loss.item()/(i_batch + 1), regularaztion_W.item()/(i_batch + 1), regularaztion_phi.item()/(i_batch + 1), correct / total

def askl(parameter_dic):
    dataset = parameter_dic['dataset']
    D = parameter_dic['D']
    stationary = parameter_dic['stationary']
    back_propagation = parameter_dic['back_propagation']
    std1 = parameter_dic['std1']
    std2 = parameter_dic['std2']
    batch_size = parameter_dic['batch_size']
    T = parameter_dic['T']
    record_batch = parameter_dic['record_batch']
    loss_type = parameter_dic['loss_type']
    lambda_A = parameter_dic['lambda_A']
    lambda_B = parameter_dic['lambda_B']
    regular_type = parameter_dic['regular_type']
    learning_rate = parameter_dic['learning_rate']

    # Load data
    trainloader, validateloader, testloader, d, K = load_data(dataset, batch_size)
    if K == 1:
        loss_type = 'mse'
    elif K == 2:
        loss_type = 'hinge'
        K = 1
    else:
        loss_type = 'cross_entroy'

    # Initial spectral_measure
    spectral_measure, W  = initial_weights(d, D, K, back_propagation, std1, std2, stationary, device)

    # Define optimizer
    #optimizer = optim.SGD((spectral_measure[0], spectral_measure[1], W), lr=0.0001, momentum=0.9)
    if stationary :
        optimizer = optim.Adam((spectral_measure[0], W), learning_rate)
    else:
        optimizer = optim.Adam((spectral_measure[0], spectral_measure[1], W), learning_rate)

    # Records variables
    training_loss_records, training_regularaztion_W_records, training_regularaztion_phi_records = [], [], []
    validate_loss_records, validate_regularaztion_W_records, validate_regularaztion_phi_records, validate_accuracy_records = [], [], [], []

    start = time.time()
    # Training
    for epoch in range(T):
        training_loss, training_regularaztion_W, training_regularaztion_phi = [0.0, 0.0, 0.0]
        for i_batch, train_batch in enumerate(trainloader, 0):
            optimizer.zero_grad() 
            X_train, y_train = train_batch
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # Forward : predict 
            phi = feature_mapping(X_train, spectral_measure, stationary)
            y_pred = phi.mm(W)
            
            # Forward : calculate objective
            loss = empirical_loss(y_pred, y_train, loss_type)
            if regular_type == 'fro':
                regularaztion_W = lambda_A * torch.norm(W, 'fro')
                objective = loss + regularaztion_W
            else:
                with torch.no_grad():
                    regularaztion_W = lambda_A * sum(torch.svd(W)[1]) #torch.norm(W, regular_type)
                regularaztion_phi = lambda_B * torch.norm(phi, 'fro')
                objective = loss + regularaztion_W + regularaztion_phi
            
            # Records
            training_loss += loss.item()
            training_regularaztion_W += regularaztion_W.item()
            training_regularaztion_phi += 0 if regular_type == 'fro' else regularaztion_phi.item()
            if i_batch % record_batch == record_batch - 1:
                training_loss_records.append(training_loss / record_batch)
                training_regularaztion_W_records.append(training_regularaztion_W / record_batch)
                training_regularaztion_phi_records.append(training_regularaztion_phi / record_batch)
                training_loss, training_regularaztion_W, training_regularaztion_phi = [0.0, 0.0, 0.0]

                if parameter_dic['validate']:
                    validate_loss, validate_regularaztion_W, validate_regularaztion_phi, validate_accuracy = test_in_batch(validateloader, spectral_measure, stationary, W, loss_type, lambda_A, lambda_B, regular_type, device)
                    validate_loss_records.append(validate_loss) 
                    validate_regularaztion_W_records.append(validate_regularaztion_W)
                    validate_regularaztion_phi_records.append(validate_regularaztion_phi) 
                    validate_accuracy_records.append(validate_accuracy)
                    print('[%d, %5d] loss: %.3f, accuracy: %.3f%%' % (epoch + 1, i_batch + 1, validate_loss, validate_accuracy))
                else:
                    print('[%d, %5d] loss: %.3f%%' % (epoch + 1, i_batch + 1, training_loss))

            # Backward
            objective.backward()
            if regular_type != 'fro':
                with torch.no_grad():
                    U, S, V = torch.svd(W)
                    W = U.mm(torch.diag(S - optimizer.defaults['lr']*lambda_A)).mm(V.T)

            # Update
            optimizer.step()
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))

    # Testing
    test_loss, test_regularaztion_W, test_regularaztion_phi, test_accuracy = test_in_batch(testloader, spectral_measure, stationary, W, loss_type, lambda_A, lambda_B, regular_type)
    if loss_type == 'mse':
        print('MSE of ASKL on the %d test examples: %.3f ' % (testloader.batch_size,test_accuracy))
        test_accuracy = - test_accuracy # For standard
    else:
        print('Accuracy of ASKL on the %d test examples: %.3f %%' % (testloader.batch_size,
            test_accuracy))

    # Save results
    result_dict = {
        'parameter_dic' : parameter_dic,
        # 'spectral_measure' : spectral_measure,
        # 'W' : W,
        'training_objective_records' : [training_loss_records[i] + training_regularaztion_W_records[i] + training_regularaztion_phi_records[i] for i in range(len(training_loss_records))],      
        'training_loss_records' : training_loss_records,
        'training_regularaztion_W_records' : training_regularaztion_W_records,
        'training_regularaztion_phi_records' : training_regularaztion_phi_records,

        'validate_objective_records' : [validate_loss_records[i] + validate_regularaztion_W_records[i] +         validate_regularaztion_W_records[i] for i in range(len(validate_loss_records))],
        'validate_loss_records' : validate_loss_records, 
        'validate_regularaztion_W_records' : validate_regularaztion_W_records,
        'validate_regularaztion_phi_records' : validate_regularaztion_W_records,
        'validate_accuracy_records' : validate_accuracy_records,

        'test_objective' : test_loss + test_regularaztion_W + test_regularaztion_phi,
        'test_loss' : test_loss,
        'test_regularaztion_W' : test_regularaztion_W,
        'test_regularaztion_phi' : test_regularaztion_phi,
        'test_accuracy' : test_accuracy,
        'training_time' : end - start
    }

    return result_dict