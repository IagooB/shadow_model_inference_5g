import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from logg import logger
from configuracion import exper_config
import numpy as np


def create_global_model(input_shape):
    """
    Crea un modelo de red neuronal para aprendizaje federado en PyTorch.

    Parameters:
        input_shape (int): Número de características de entrada.

    Returns:
        torch.nn.Module: Modelo de PyTorch.
    """

    class GlobalModel(nn.Module):
        def __init__(self, input_shape):
            super(GlobalModel, self).__init__()
            self.fc1 = nn.Linear(input_shape, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.sigmoid(self.fc3(x))
            return x

    logger.info(f"Creando modelo global con input shape {input_shape}")
    return GlobalModel(input_shape)


def create_shadow_model(input_shape):
    """
    Crea un modelo sombra para inferencia de propiedades en PyTorch.

    Parameters:
        input_shape (int): Dimensión de entrada.

    Returns:
        torch.nn.Module: Modelo de PyTorch.
    """

    class ShadowModel(nn.Module):
        def __init__(self, input_shape):
            super(ShadowModel, self).__init__()
            self.fc1 = nn.Linear(input_shape, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.sigmoid(self.fc3(x))
            return x

    logger.info(f"Creando modelo sombra con input shape {input_shape}")
    return ShadowModel(input_shape)

class AttackModelNN(nn.Module):
    def __init__(self, input_dim):
        super(AttackModelNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        # Si deseas más capas, agrégalas.

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sale prob en [0,1]
        return x

    def train_model(self, X_train, y_train, epochs, lr, batch_size):
        """
        Entrena el modelo AttackModelNN en un dataset (X_train, y_train) con PyTorch.
        :param X_train: np.array con shape [N, input_dim]
        :param y_train: np.array con shape [N]
        :param epochs: número de épocas
        :param lr: learning rate
        :param batch_size: batch size
        """
        # 1) Convertir a tensores
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.train()  # modo train
        for ep in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                out = self.forward(batch_x).squeeze(dim=1)  # [BS]
                loss = loss_fn(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {ep+1}/{epochs}, Loss={avg_loss:.6f}")

    def predict_proba(self, X_input_np):
        """
        Dado un np.array shape (N, input_dim),
        retorna probabilities shape (N, 2) al estilo scikit-learn
        donde la 2a columna es la prob. de clase=1
        """
        self.eval()
        X_t = torch.tensor(X_input_np, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.forward(X_t).squeeze(dim=1)  # shape (N,)
            # outputs es prob de la clase=1
            prob_1 = outputs.detach().cpu().numpy()
        prob_0 = 1.0 - prob_1
        # unimos [prob_0, prob_1] en shape (N,2)
        probs = np.stack([prob_0, prob_1], axis=1)
        return probs



def train_shadow_models_sin_epoch(X_shadow, y_label_shadow, y_slice_shadow, global_model):
    """
    Entrena varios modelos sombra, separando los datos con propiedad (Slice=1)
    y sin propiedad (Slice=0). Además, recopila actualizaciones etiquetadas
    para entrenar luego un Attack Model.

    Return:
      shadow_models (list): la lista de shadow models entrenados.
      X_attack (list of np arrays): lista de vectores de actualización etiquetados
      y_attack (list of int): etiquetas 0/1 indicando si el vector de actualización
                              corresponde a 'prop=0' o 'prop=1'
    """
    logger.info(f"Training {exper_config['num_shadow_models']} shadow models prior to federated learning.")

    shadow_models = []
    X_attack = []
    y_attack = []

    # Separamos los datos de shadow en 2 subconjuntos
    X_prop = X_shadow[y_slice_shadow == 1]
    y_prop = y_label_shadow[y_slice_shadow == 1]

    X_noprop = X_shadow[y_slice_shadow == 0]
    y_noprop = y_label_shadow[y_slice_shadow == 0]

    # Suponemos que exper_config['num_shadow_models'] es par,
    # la mitad entrenan en "prop" y la mitad en "noprop":
    num_sm = exper_config['num_shadow_models']
    half = num_sm // 2

    loss_fn = nn.BCELoss()

    # Entrenamos half shadow models con "prop=1"
    for i in range(half):
        logger.info(f"Training shadow model (PROP) {i+1}/{half}")

        shadow_model = type(global_model)(global_model.fc1.in_features)
        shadow_model.load_state_dict(global_model.state_dict())
        shadow_model.train()

        optimizer = optim.Adam(shadow_model.parameters(), lr=exper_config["lr_shadow_training"])

        # Entrenamiento sencillo
        for epoch in range(exper_config['shadow_train_rounds']):
            optimizer.zero_grad()
            out = shadow_model(X_prop)
            loss = loss_fn(out.squeeze(), y_prop)
            loss.backward()
            optimizer.step()

        shadow_models.append(shadow_model)

        # Guardar vector de actualización final
        final_sd = shadow_model.state_dict()
        update_vec_list = []
        for k, v in final_sd.items():
            update_vec_list.append(v.flatten())
        merged_vec = torch.cat(update_vec_list, dim=0)

        X_attack.append( merged_vec.detach().cpu().numpy() )
        y_attack.append(1)  # 1 => propiedad presente

    # Entrenamos half shadow models con "prop=0"
    for i in range(half):
        logger.info(f"Training shadow model (NOPROP) {i+1}/{half}")

        shadow_model = type(global_model)(global_model.fc1.in_features)
        shadow_model.load_state_dict(global_model.state_dict())
        shadow_model.train()

        optimizer = optim.Adam(shadow_model.parameters(), lr=exper_config["lr_shadow_training"])

        for epoch in range(exper_config['shadow_train_rounds']):
            optimizer.zero_grad()
            out = shadow_model(X_noprop)
            loss = loss_fn(out.squeeze(), y_noprop)
            loss.backward()
            optimizer.step()

        shadow_models.append(shadow_model)

        # Guardar vector de actualización final
        final_sd = shadow_model.state_dict()
        update_vec_list = []
        for k, v in final_sd.items():
            update_vec_list.append(v.flatten())
        merged_vec = torch.cat(update_vec_list, dim=0)

        X_attack.append( merged_vec.detach().cpu().numpy() )
        y_attack.append(0)  # 0 => sin propiedad

    logger.info("All shadow models trained successfully. Building X_attack, y_attack dataset.")
    return shadow_models, X_attack, y_attack


def train_shadow_models(
        X_shadow,
        y_label_shadow,
        y_slice_shadow,
        global_model,
        shadow_train_rounds=10
):
    """
    Entrena varios modelos sombra, separando los datos con propiedad (Slice=1)
    y sin propiedad (Slice=0). Además, en cada epoch se guarda el vector de
    actualización, para tener más datos al entrenar el modelo de ataque.

    Return:
      shadow_models (list): Lista de shadow models entrenados.
      X_attack (list): Lista de vectores de actualización etiquetados (in/out).
      y_attack (list): Etiquetas 0/1 indicando si el vector pertenece a 'prop=1' o 'prop=0'.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    logger.info(f"Training {exper_config['num_shadow_models']} shadow models with epoch-level updates.")

    shadow_models = []
    X_attack = []
    y_attack = []

    # Separamos datos en prop=1 y prop=0
    X_prop = X_shadow[y_slice_shadow == 1]
    y_prop = y_label_shadow[y_slice_shadow == 1]

    X_noprop = X_shadow[y_slice_shadow == 0]
    y_noprop = y_label_shadow[y_slice_shadow == 0]

    num_sm = exper_config['num_shadow_models']
    half = num_sm // 2  # mitad para 'prop=1', mitad para 'prop=0'

    loss_fn = nn.BCELoss()

    # -------------------------------------
    # 1) Entrenamos 'half' shadow models con 'prop=1'
    # -------------------------------------
    for i in range(half):
        logger.info(f"Training shadow model (PROP) {i + 1}/{half}")

        # Creamos un nuevo modelo sombra con la misma estructura que el global
        shadow_model = type(global_model)(global_model.fc1.in_features)
        shadow_model.load_state_dict(global_model.state_dict())
        shadow_model.train()

        optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)

        # Para calcular actualizaciones por epoch
        prev_sd = {}
        for k, v in shadow_model.state_dict().items():
            prev_sd[k] = v.clone().detach()

        for epoch in range(shadow_train_rounds):
            optimizer.zero_grad()
            out = shadow_model(X_prop)
            loss = loss_fn(out.squeeze(), y_prop)
            loss.backward()
            optimizer.step()

            # Guardar vector de actualización de esta epoch
            current_sd = shadow_model.state_dict()
            # vector de diferencia epoch_i = current - prev
            update_vec_list = []
            for k, v in current_sd.items():
                diff = v - prev_sd[k]
                update_vec_list.append(diff.flatten())
            merged_vec = torch.cat(update_vec_list, dim=0)

            X_attack.append(merged_vec.detach().cpu().numpy())
            # Este update es 'prop=1'
            y_attack.append(1)

            # Actualizar prev_sd para la siguiente epoch
            for k, v in current_sd.items():
                prev_sd[k] = v.clone().detach()

        shadow_models.append(shadow_model)

    # -------------------------------------
    # 2) Entrenamos 'half' shadow models con 'prop=0'
    # -------------------------------------
    for i in range(half):
        logger.info(f"Training shadow model (NOPROP) {i + 1}/{half}")

        shadow_model = type(global_model)(global_model.fc1.in_features)
        shadow_model.load_state_dict(global_model.state_dict())
        shadow_model.train()

        optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)

        # Para calcular actualizaciones por epoch
        prev_sd = {}
        for k, v in shadow_model.state_dict().items():
            prev_sd[k] = v.clone().detach()

        for epoch in range(shadow_train_rounds):
            optimizer.zero_grad()
            out = shadow_model(X_noprop)
            loss = loss_fn(out.squeeze(), y_noprop)
            loss.backward()
            optimizer.step()

            # Guardar vector de actualización de esta epoch
            current_sd = shadow_model.state_dict()
            update_vec_list = []
            for k, v in current_sd.items():
                diff = v - prev_sd[k]
                update_vec_list.append(diff.flatten())
            merged_vec = torch.cat(update_vec_list, dim=0)

            X_attack.append(merged_vec.detach().cpu().numpy())
            # Este update es 'prop=0'
            y_attack.append(0)

            for k, v in current_sd.items():
                prev_sd[k] = v.clone().detach()

        shadow_models.append(shadow_model)

    logger.info("All shadow models trained. Built X_attack, y_attack with epoch-level updates.")
    return shadow_models, X_attack, y_attack
