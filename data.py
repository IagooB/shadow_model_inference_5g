import torch
from logg import logger
from configuracion import *
import numpy as np
import pandas as pd
import os
from main import set_seed

set_seed()


def load_data(file_path):
    """
    Carga un archivo CSV y devuelve un DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no se encontró.")

    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, fraction=exper_config["fraction"]):
    """
    Preprocesa los datos, filtra columnas necesarias y aplica muestreo.
    """
    necessary_columns = [
        'Src IP', 'Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Total Fwd Packet',
        'Fwd Packet Length Std', 'ACK Flag Count', 'Fwd Seg Size Min', 'label', 'Slice'
    ]

    df = df[necessary_columns].dropna()

    if exper_config["aleatoriedad"]:
        df = df.sample(frac=fraction).reset_index(drop=True)
    else:
        df = df.sample(frac=fraction, random_state=SEED).reset_index(drop=True)

    X = df.drop(['label', 'Slice'], axis=1).values  # Convertir a numpy
    y_label = df['label'].values  # Convertir a numpy
    y_slice = df['Slice'].values  # Convertir a numpy

    return X, y_label, y_slice


def create_client_data(X, y_label, y_slice):
    """
    Divide los datos para los clientes y aplica ruido o flipping si está configurado.
    """
    num_clients = exper_config["num_clients"]
    logger.info(f"Creating data for {num_clients} clients")

    X_with_property = X[y_slice == 1]
    y_label_with_property = y_label[y_slice == 1]
    X_without_property = X[y_slice == 0]
    y_label_without_property = y_label[y_slice == 0]

    if num_clients % 2 != 0:
        raise ValueError("El número de clientes debe ser par para balancear propiedades.")

    min_data_size = min(len(X_with_property) // (num_clients // 2), len(X_without_property) // (num_clients // 2))

    client_data = []
    for i in range(num_clients // 2):
        client_data.append({
            'X': torch.tensor(X_with_property[i * min_data_size:(i + 1) * min_data_size], dtype=torch.float32),
            'y_label': torch.tensor(y_label_with_property[i * min_data_size:(i + 1) * min_data_size],
                                    dtype=torch.float32),
            'y_slice': 1,
            'has_property': True
        })
        client_data.append({
            'X': torch.tensor(X_without_property[i * min_data_size:(i + 1) * min_data_size], dtype=torch.float32),
            'y_label': torch.tensor(y_label_without_property[i * min_data_size:(i + 1) * min_data_size],
                                    dtype=torch.float32),
            'y_slice': 0,
            'has_property': False
        })

    return client_data


def split_data_for_models(X, y_label, y_slice):
    """
    Divide los datos en conjuntos para el modelo global y los modelos sombra.
    """
    logger.info("Dividiendo datos para el modelo global y los modelos sombra.")

    num_shadow_samples = int(len(X) * exper_config['shadow_data_fraction'])
    indices = np.arange(len(X))

    if exper_config["aleatoriedad"]:
        np.random.shuffle(indices)

    shadow_indices = indices[:num_shadow_samples]
    global_indices = indices[num_shadow_samples:]

    X_shadow, y_label_shadow, y_slice_shadow = X[shadow_indices], y_label[shadow_indices], y_slice[shadow_indices]
    X_global, y_label_global, y_slice_global = X[global_indices], y_label[global_indices], y_slice[global_indices]

    logger.info(f"Datos divididos: {len(X_global)} para el modelo global, {len(X_shadow)} para los modelos sombra.")

    return (X_global, y_label_global, y_slice_global), (X_shadow, y_label_shadow, y_slice_shadow), len(X_shadow)
