from models import *
from concurrent.futures import ThreadPoolExecutor
from logg import *
import pandas as pd


def select_clients(round_num, clients_with_property, clients_without_property):
    """
    Selecciona los clientes de forma determinista:
    - En rondas pares: Se eligen clientes con la propiedad.
    - En rondas impares: Se eligen clientes sin la propiedad.
    """
    return clients_with_property if round_num % 2 == 0 else clients_without_property


def initialize_clients(client_data, global_model, global_model_epochs, batch_size):
    def init_client(client_id, data):
        # Pasar el modelo global completo
        return SimulatedFlowerClient(client_id, data, global_model, global_model_epochs, batch_size)

    # Paralelizar la inicialización de los clientes
    if exper_config["aleatoriedad"]:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(init_client, i, data) for i, data in enumerate(client_data)]
            clients = [f.result() for f in futures]
    else:
        clients = [init_client(i, data) for i, data in enumerate(client_data)]

    return clients


class SimulatedFlowerClient:
    def __init__(self, client_id, data, model, config, batch_size):
        """
        Inicializa un cliente simulado con sus datos y modelo en Federated Learning.
        """
        self.client_id = client_id
        self.data = data
        self.X = data["X"].clone().detach() if isinstance(data["X"], torch.Tensor) else torch.tensor(data["X"].values,
                                                                                                     dtype=torch.float32) if isinstance(
            data["X"], pd.DataFrame) else torch.tensor(data["X"], dtype=torch.float32)
        self.y_label = data["y_label"].clone().detach() if isinstance(data["y_label"], torch.Tensor) else torch.tensor(
            data["y_label"], dtype=torch.float32)
        self.config = config
        self.batch_size = batch_size

        # Inicializar el modelo copiando el global
        self.model = model
        self.model.load_state_dict(model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=exper_config["lr_cliente_simulado"])
        self.loss_fn = nn.BCELoss()

    def get_parameters(self):
        """Devuelve los parámetros actuales del modelo del cliente."""
        return self.model.state_dict()

    def fit(self, global_weights):
        """
        Entrena el modelo localmente y devuelve las actualizaciones.
        """
        self.model.load_state_dict(global_weights)
        self.model.train()

        for epoch in range(self.config["global_model_epochs"]):
            self.optimizer.zero_grad()
            outputs = self.model(self.X)
            loss = self.loss_fn(outputs.squeeze(), self.y_label)
            loss.backward()
            self.optimizer.step()

        new_weights = self.model.state_dict()
        updates = {}
        for key in new_weights.keys():
            gw = global_weights[key]
            # si es np array, conviene cast a torch
            if not isinstance(gw, torch.Tensor):
                gw = torch.tensor(gw, dtype=torch.float32)

            updates[key] = new_weights[key] - gw.clone().detach()

        return new_weights, len(self.X), {"updates": updates}

    def evaluate(self, global_weights):
        """
        Evalúa el modelo global en los datos locales del cliente.
        """
        self.model.load_state_dict(global_weights)
        self.model.eval()
        with torch.no_grad():
            X_input = self.X if isinstance(self.X, torch.Tensor) else torch.tensor(self.X, dtype=torch.float32)
            outputs = self.model(X_input)

            loss = self.loss_fn(outputs.squeeze(), self.y_label)
            accuracy = ((outputs.squeeze() > 0.5) == self.y_label).float().mean().item()

        return loss.item(), len(self.X), {"accuracy": accuracy}
