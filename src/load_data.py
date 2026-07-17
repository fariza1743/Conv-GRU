# This file is used to load the data from pickle files into TensorDataset format.

import torch
from torch.utils.data import TensorDataset
import numpy as np
import pickle


def _smart_load(filename: str):
    """
    Robust loader that can handle:
      - torch.save(...) files (often trigger pickle persistent_id errors if loaded with pickle)
      - plain pickle files

    Returns the loaded Python object.
    """
    # 1) Try torch.load first (fixes persistent ID errors for torch-saved files)
    try:
        return torch.load(filename, map_location="cpu")
    except Exception:
        pass

    # 2) Fallback to normal pickle load
    with open(filename, "rb") as fd:
        return pickle.load(fd)


def load_data(filename, pretrain=False, patient=None, unrestricted=True, combined=False):
    obj = _smart_load(filename)

    # Your original code expected the file to contain a 2-tuple:
    #   if pretrain: (_, data) = ...
    #   else:        (data, _) = ...
    if not (isinstance(obj, (tuple, list)) and len(obj) == 2):
        raise ValueError(
            f"Unexpected file format in {filename}. "
            f"Expected a 2-item tuple/list, got type={type(obj)} with value structure not matching."
        )

    if pretrain:
        (_, data) = obj
    else:
        (data, _) = obj

    # --- rest of your original logic unchanged ---
    if pretrain:
        scaling = data['scaling']
    else:
        scaling = data['scaling'][patient]

    (training_x, training_y) = data['training']
    (validation_x, validation_y) = data['validation']
    (testing_x, testing_y) = data['testing']

    case = '3' if unrestricted else '1'

    if pretrain:
        tr_x = training_x[case]['input1_layer']
        tr_x2 = training_x[case]['input3_layer']
        tr_x3 = training_x[case]['input2_layer']
        tr_y = training_y[case]

        va_x = validation_x[case]['input1_layer']
        va_x2 = validation_x[case]['input3_layer']
        va_x3 = validation_x[case]['input2_layer']
        va_y = validation_y[case]

        te_x = testing_x[case]['input1_layer']
        te_x2 = testing_x[case]['input3_layer']
        te_x3 = testing_x[case]['input2_layer']
        te_y = testing_y[case]

    else:
        tr_x = training_x[patient][case]['input1_layer']
        tr_x2 = training_x[patient][case]['input3_layer']
        tr_x3 = training_x[patient][case]['input2_layer']
        tr_y = training_y[patient][case]

        va_x = validation_x[patient][case]['input1_layer']
        va_x2 = validation_x[patient][case]['input3_layer']
        va_x3 = validation_x[patient][case]['input2_layer']
        va_y = validation_y[patient][case]

        te_x = testing_x[patient][case]['input1_layer']
        te_x2 = testing_x[patient][case]['input3_layer']
        te_x3 = testing_x[patient][case]['input2_layer']
        te_y = testing_y[patient][case]

    if combined:
        print('COMBINED')
        tr_x = np.append(tr_x, va_x, axis=0)
        tr_x2 = np.append(tr_x2, va_x2, axis=0)
        tr_x3 = np.append(tr_x3, va_x3, axis=0)
        tr_y = np.append(tr_y, va_y, axis=0)

    training_dataset = TensorDataset(
        torch.from_numpy(tr_x).type(torch.float),
        torch.from_numpy(tr_x2).type(torch.float),
        torch.from_numpy(tr_x3).type(torch.float),
        torch.from_numpy(tr_y).unsqueeze(1).type(torch.float)
    )

    testing_dataset = TensorDataset(
        torch.from_numpy(te_x).type(torch.float),
        torch.from_numpy(te_x2).type(torch.float),
        torch.from_numpy(te_x3).type(torch.float),
        torch.from_numpy(te_y).unsqueeze(1).type(torch.float)
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(va_x).type(torch.float),
        torch.from_numpy(va_x2).type(torch.float),
        torch.from_numpy(va_x3).type(torch.float),
        torch.from_numpy(va_y).unsqueeze(1).type(torch.float)
    )

    return ((training_dataset, testing_dataset, validation_dataset), scaling)
