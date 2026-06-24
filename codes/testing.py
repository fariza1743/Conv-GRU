import torch
from torch.utils.data import DataLoader
import numpy as np
from load_data import load_data
import json
import argparse
import matplotlib.pyplot as plt
	
def RMSE(predictions, labels):
	rmse = (predictions - labels) ** 2
	rmse = np.mean(rmse)
	rmse = np.sqrt(rmse)
	return rmse
	
	
def MAE(predictions, labels):
	mae = np.abs(predictions - labels)
	mae = np.mean(mae)
	return mae

def evaluate(data, 
             patient_list,
             scaling,
             batch_size,
             testing_set='validation', 
             unrestricted=True, 
             runs=1, 
             tensorboard=False,
             return_traces=False,              # <--- NEW
             plot_selected=True):              # <--- NEW
    ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack scaling parameters if provided
    try:
        if scaling is None:
            scale_min, scale_max = 0.0, 1.0
        elif isinstance(scaling, (list, tuple)) and len(scaling) == 2:
            scale_min, scale_max = scaling
        elif isinstance(scaling, dict):
            # Try common key names
            if "min" in scaling and "max" in scaling:
                scale_min, scale_max = scaling["min"], scaling["max"]
            elif "scale_min" in scaling and "scale_max" in scaling:
                scale_min, scale_max = scaling["scale_min"], scaling["scale_max"]
            elif "data_min_" in scaling and "data_max_" in scaling:  # sklearn MinMaxScaler
                scale_min, scale_max = scaling["data_min_"], scaling["data_max_"]
            else:
                # fallback if dictionary contains nested patient-level data
                first_key = list(scaling.keys())[0]
                inner = scaling[first_key]
                scale_min = inner.get("min", inner.get("scale_min", 0.0))
                scale_max = inner.get("max", inner.get("scale_max", 1.0))
        else:
            raise TypeError("Unsupported scaling format.")
    except Exception:
        scale_min, scale_max = 0.0, 1.0  # final fallback to prevent crash

    scale_range = scale_max - scale_min

    # Load data for each patient
    #train_loader, validation_loader, testing_loader = load_data(data, pretrain=False, patient=None, unrestricted=True, combined=False)
    results = {}
    best_models = {}
    traces = {}                                # <--- NEW
    for patient in patient_list:
       
        #-----------------------------------------------------------------------------------------------------------------------  
        (datasets, patient_scaling) = load_data(data, pretrain=False, patient=patient, unrestricted=unrestricted, combined=False)
        training_dataset, testing_dataset, validation_dataset = datasets
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
        testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        #-----------------------------------------------------------------------------------------------------------------------  
        ...
        best_mae = np.inf
        best_models[patient] = 0

        # --- pass 1: find best run on validation ---
        for run in range(runs): 
            network = torch.load(f'./patient_{patient}_{run}.model', weights_only=False).get('model').cuda()
            network.eval()
            predictions, labels = [], []

            for step, (vx, vx2, vx3, vy) in enumerate(validation_loader):
                with torch.no_grad():
                    if unrestricted:
                        prediction = network(vx.to(device), vx2.to(device), x3=vx3.to(device))
                    else:
                        prediction = network(vx.to(device), vx2.to(device))
                predictions += prediction.cpu().data.numpy().tolist()
                labels += vy.cpu().data.numpy().tolist()

            predictions = np.asarray(predictions)
            labels = np.asarray(labels)
            predictions = predictions * scale_range + scale_min
            labels = labels * scale_range + scale_min
            mae = MAE(predictions, labels)
            if mae < best_mae:
                best_mae = mae
                best_models[patient] = run

        results[patient] = {'RMSEs': [], 'MAEs': []}
        if return_traces:
            traces[patient] = {}               # <--- NEW

        # --- pass 2: evaluate on TEST set; optionally plot the best run ---
        for run in range(runs):
            np.random.seed(run)
            torch.manual_seed(run)
            network = torch.load(f'./patient_{patient}_{run}.model', weights_only=False).get('model').cuda()
            network.eval()
            predictions, labels = [], []

            for step, (x, x2, x3, y) in enumerate(testing_loader):
                with torch.no_grad():
                    if unrestricted:
                        prediction = network(x.to(device), x2.to(device), x3=x3.to(device))
                    else:
                        prediction = network(x.to(device), x2.to(device))
                predictions += prediction.cpu().data.numpy().tolist()
                labels += y.cpu().data.numpy().tolist()

            predictions = np.asarray(predictions)
            labels = np.asarray(labels)
            predictions = predictions * scale_range + scale_min
            labels = labels * scale_range + scale_min

            rmse = RMSE(predictions, labels)
            mae  = MAE(predictions, labels)
            results[patient]['RMSEs'].append(rmse)
            results[patient]['MAEs'].append(mae)

            if return_traces:
                traces[patient][run] = {'y': labels, 'yhat': predictions}  # <--- NEW

            # Plot only the selected/best run for this patient
            if plot_selected and run == best_models[patient]:
                plt.figure(figsize=(10, 4))
                plt.plot(labels, label='True', linewidth=2)
                plt.plot(predictions, label='Predicted', linestyle='--', linewidth=2)
                plt.title(f'Patient {patient} - Best Run {run} (Test set)')
                plt.xlabel('Sample Index')
                plt.ylabel('Value')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'patient_{patient}_best_run_{run}_pred_vs_true.png')
                plt.close()

    if return_traces:
        return (results, best_models, traces)   # <--- NEW
    return (results, best_models)

import numpy as np

def compute_final_results(results, best_models, patient_list):

    bests = {}
    averages = {}
    stds = {}

    # ---------- Per-patient statistics ----------
    for p in patient_list:
        rmses = np.array(results[p]['RMSEs'])
        maes  = np.array(results[p]['MAEs'])

        # Best model values
        idx = best_models[p]
        bests[p] = {
            'RMSE': rmses[idx],
            'MAE':  maes[idx]
        }

        # Mean values
        averages[p] = {
            'RMSE': rmses.mean(),
            'MAE':  maes.mean()
        }

        # Standard deviations
        stds[p] = {
            'RMSE': rmses.std(ddof=1),   # sample standard deviation
            'MAE':  maes.std(ddof=1)
        }

    # ---------- Overall statistics ----------
    avg_rmse_all = np.array([averages[p]['RMSE'] for p in patient_list])
    avg_mae_all  = np.array([averages[p]['MAE'] for p in patient_list])
    best_rmse_all = np.array([bests[p]['RMSE'] for p in patient_list])
    best_mae_all  = np.array([bests[p]['MAE'] for p in patient_list])
    std_rmse_all  = np.array([stds[p]['RMSE'] for p in patient_list])
    std_mae_all   = np.array([stds[p]['MAE'] for p in patient_list])

    overall = {
        'AVG': {
            'RMSE': avg_rmse_all.mean(),
            'MAE':  avg_mae_all.mean()
        },
        'BEST': {
            'RMSE': best_rmse_all.mean(),
            'MAE':  best_mae_all.mean()
        },
        'STD': {
            'RMSE': std_rmse_all.mean(),
            'MAE':  std_mae_all.mean()
        }
    }

    by_patient = {
        'AVG': averages,
        'BEST': bests,
        'STD': stds
    }

    return {'by_patient': by_patient, 'overall': overall}
