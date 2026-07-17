import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def getplotvalue(
    data,
    patient_list,
    scaling,
    batch_size,
    testing_set='validation',
    unrestricted=True,
    runs=1,
    tensorboard=False
):
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictions, labels = None, None  # will hold last patient/last run values

    for patient in patient_list:
        (training, testing, validation) = data[patient]

        if testing_set == 'training':
            dataset = training
        elif testing_set == 'validation':
            dataset = validation
        elif testing_set == 'testing':
            dataset = testing
        else:
            raise ValueError(f"Unknown testing_set: {testing_set}")

        # For plotting/sequence integrity, prefer no shuffle
        
        validation_loader = DataLoader(dataset=validation, shuffle=False, drop_last=False)

        scale_min = scaling[patient]['min']
        scale_max = scaling[patient]['max']

        # only produce the last run’s arrays; adjust if you want best run instead
        run = runs - 1
        state = torch.load(f'./patient_{patient}_{run}.model', map_location=device)
        network = state.get('model').to(device)
        network.eval()

        preds, labs = [], []
        with torch.no_grad():
            for batch in validation_loader:
                if unrestricted:
                    vx, vx2, vx3, vy = batch
                    out, *_ = network(vx.to(device), vx2.to(device), x3=vx3.to(device))
                else:
                    vx, vx2, vy = batch  # adjust if your inertial dataset yields 3 items
                    out, *_ = network(vx.to(device), vx2.to(device))
                preds += out.cpu().numpy().tolist()
                labs  += vy.cpu().numpy().tolist()

        preds = np.asarray(preds) * scale_max + scale_min
        labs  = np.asarray(labs)  * scale_max + scale_min

        # keep the last patient’s last run results
        predictions, labels = preds, labs

    return predictions, labels