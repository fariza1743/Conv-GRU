#!/home/jb199113/anaconda3/bin/python

# This file contains the code to train models that are specific to a particular subject.
# The finetune function loads a pretrained model and trains it further on a particular
# subjects data.

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json

def RMSE(predictions, labels):
    rmse = (predictions - labels) ** 2
    rmse = np.mean(rmse)
    rmse = np.sqrt(rmse)
    return rmse
    
    
def MAE(predictions, labels):
    mae = np.abs(predictions - labels)
    mae = np.mean(mae)
    return mae
    
# This is the main function for finetuning subject specific models.
#
# Parameters:
#    data = Data that is in TensorDataset formats. The data is already formatted
#   properly if the load_data function from the load_data.py file was used.
#
#    hyper_parameters = A dictonary of hyper_parameters. For an example of the format,
#    look at the hyperparameters subdirectory
#
#    patient_list = a list of the patient ids that should have models finetuned for them
#
#    unrestricted = true if unrestricted case, false if inertial
#
#    runs = the number of runs for each subject
#
#    tensorboard = true if tensorboard logs should be generated
#
#    track_epochs = true if you would like to save a file with information on the
#   amount of epochs trained prior to early stopping
#
#    specific_epochs = If you wish to specify the number of epochs to train, supply a
#     dictonary with entries in this format: {SubjectID: num_epochs}. Set this to None
#    If you do not wish to specify the number of epochs.

def finetune(data, 
             hyper_parameters, 
             patient_list, 
             unrestricted=True, 
             runs=1, 
             tensorboard=False,
             track_epochs=False,
             specific_epochs=None):
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    batch_size = hyper_parameters['batch_size']
    patience = hyper_parameters['patience']
    
    epochs = {}
    
    for patient in patient_list:
        (training, _, validation) = data[patient]
        
        training_loader = DataLoader(dataset=training,
                                     batch_size=batch_size,
                                     shuffle=True)
                                
        validation_loader = DataLoader(dataset=validation,
                                       batch_size=batch_size,
                                       shuffle=True)

        epochs[patient] = {}
        for run in range(runs):
            np.random.seed(run)
            torch.manual_seed(run)
        
            network = torch.load('./base_best.model', weights_only=False).get('model').cuda()
                              
            optimizer = optim.Adam(network.parameters(), lr=hyper_parameters['learning_rate'])
            
            loss_function = nn.MSELoss()
            
            if tensorboard:
                writer = SummaryWriter('tensorboard/' + patient + '_' + str(run) + '_' + str(hyper_parameters['learning_rate']))
                
            training_loss = []
            validation_loss = []
            
            min_validation_loss = np.inf
            unimproved = 0
            
            epochs[patient][run] = {'val_loss': [], 'selected_epoch': 0}
            num_epochs = hyper_parameters['epochs']
            if specific_epochs is not None:
                if isinstance(specific_epochs, dict):
                    num_epochs = specific_epochs[patient]
                else:
                    num_epochs = specific_epochs
                
            for e in range(num_epochs):
                network.train()
                
                total_training_loss = []
                total = 0
                
                for step, (x, x2, x3, y) in enumerate(training_loader):
                    total = total + x.shape[0]
                    optimizer.zero_grad()
                    
                    if unrestricted:
                        forecast = network(x.to(device), x2.to(device), x3=x3.to(device))
                    else:
                        forecast = network(x.to(device), x2.to(device))
                    
                    loss = loss_function(forecast, y.to(device))
                    loss.backward()
                    optimizer.step()
                    
                    total_training_loss.append(loss.item() * x.shape[0])
                    
                training_loss.append(np.sum(total_training_loss) / total)
        
                if tensorboard:
                    writer.add_scalar('training_loss', training_loss[-1], e)
        
                network.eval()
        
                total_validation_loss = []
                total = 0
        
                for step, (vx, vx2, vx3, vy) in enumerate(validation_loader):
                    total = total + vx.shape[0]
                    
                    with torch.no_grad():
                    
                        if unrestricted:
                            forecast = network(vx.to(device), vx2.to(device), x3=vx3.to(device))
                        else:
                            forecast = network(vx.to(device), vx2.to(device))
        
                    loss = loss_function(forecast, vy.to(device))

                    total_validation_loss.append(loss.item() * vx.shape[0])
                    
                validation_loss.append(np.sum(total_validation_loss) / total)
                
                if tensorboard:
                    writer.add_scalar('validation_loss', validation_loss[-1], e)

                if specific_epochs is None:
                    if validation_loss[-1] < min_validation_loss:
                        state = {'epoch': e,
                                 'model': network,
                                 'training_loss': training_loss,
                                 'validation_loss': validation_loss
                                }
                                
                        torch.save(state, './patient_' + patient + '_' + str(run) + '.model')
                        
                        min_validation_loss = validation_loss[-1]
                        unimproved = 0
                        
                        epochs[patient][run]['selected_epoch'] = e
                        
                    else:
                        unimproved = unimproved + 1
                        
                    epochs[patient][run]['val_loss'].append((e, validation_loss[-1]))
                        
                    if unimproved > patience:
                        break
                        
                else:
                    if validation_loss[-1] < min_validation_loss:
                        state = {'epoch': e,
                             'model': network,
                             'training_loss': training_loss,
                             'validation_loss': validation_loss
                            }
                        torch.save(state, './patient_' + patient + '_' + str(run) + '.model')
                        
                        min_validation_loss = validation_loss[-1]
                    
                print('EPOCH:', e)
                
        if track_epochs:
            fd = open('finetune_epochs.json', 'w')
            json.dump(epochs, fd)
            fd.close()
