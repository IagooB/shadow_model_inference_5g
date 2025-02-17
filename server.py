from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, f1_score, roc_auc_score, recall_score
import torch
import numpy as np
from logg import logger


def aggregate_updates(client_weights, client_sample_counts):
    """
    Realiza una agregación ponderada de los pesos del modelo en función
    del número de muestras de cada cliente.
    """
    total_samples = sum(client_sample_counts)
    aggregated_weights = {}
    for key in client_weights[0].keys():
        # sumamos w_i * (samples_i / total_samples)
        aggregated_weights[key] = sum(
            client_weights[i][key] * (client_sample_counts[i] / total_samples)
            for i in range(len(client_weights))
        )
    return aggregated_weights


class FederatedServer:
    def __init__(self, global_model, clients, attack_model, num_rounds):
        """
        :param global_model: el modelo global (PyTorch) que se entrena federadamente
        :param clients: lista de clientes
        :param attack_model: el modelo de ataque ya entrenado (p.ej. RandomForest),
                             con un método predict_proba() o predict()
        :param num_rounds: número de rondas
        """
        self.global_model = global_model
        self.clients = clients
        self.attack_model = attack_model  # <--- SE RECIBE AQUI
        self.num_rounds = num_rounds

    def train(self):
        """
        Ejecuta el proceso de aprendizaje federado en varias rondas.
        """
        logger.info(f"Starting Federated Learning with {self.num_rounds} rounds")
        results = []

        for round_num in range(1, self.num_rounds + 1):
            logger.info(f"Round {round_num} - Selecting clients for training")
            selected_clients, has_property = self.select_clients(round_num)
            logger.info(f"{len(selected_clients)} clients selected for training. Has Property: {has_property}")

            global_weights = self.global_model.state_dict()

            # 1) Recolectar actualizaciones de los clientes
            client_weights = []
            client_sample_counts = []

            for client in selected_clients:
                updated_weights, sample_count, update_info = client.fit(global_weights)
                client_weights.append(updated_weights)
                client_sample_counts.append(sample_count)

            # 2) Agregamos (average) las actualizaciones y actualizamos el global
            averaged_weights = aggregate_updates(client_weights, client_sample_counts)
            self.global_model.load_state_dict(averaged_weights)

            # 3) Construimos el vector de actualización para cada cliente
            client_updates_vectors = []
            for cw in client_weights:
                # Convertir todas las capas del cliente en un único vector
                param_list = []
                for k, v in cw.items():
                    param_list.append(v.view(-1))  # aplanar
                merged = torch.cat(param_list, dim=0)  # concatenar
                client_updates_vectors.append(merged)

            # 4) Inferencia de la propiedad usando *attack_model*
            #    (ya entrenado con las actualizaciones de los shadow models)
            property_prob = self.infer_property_with_attack(client_updates_vectors)

            # 5) Evaluación del modelo global
            result = self.evaluate(round_num)
            result.update({
                'property_probability': property_prob,
                'has_property': has_property,
                'prediction': property_prob > 0.5,
                'threshold_used': 0.5
            })
            results.append(result)

            logger.info(f"Round {round_num} - Property Probability: {property_prob}")

        return results

    def infer_property_with_attack(self, client_updates_vectors):
        """
        Aplica el modelo de ataque (AttackModel) a los vectores de actualización
        para estimar la probabilidad de que la propiedad esté presente.
        """
        if self.attack_model is None:
            logger.warning("No attack_model provided. Returning 0.5 by default.")
            return 0.5

        # Convertimos cada update a numpy y pedimos prob al attack model
        property_preds = []
        for upd in client_updates_vectors:
            # shape [vector_dim]
            upd_np = upd.cpu().numpy().reshape(1, -1)  # [1, vector_dim]
            # suponemos que attack_model tiene predict_proba
            probs = self.attack_model.predict_proba(upd_np)[0]
            prob_class_1 = probs[1]  # asumiendo 2 clases -> (clase_0, clase_1)
            property_preds.append(prob_class_1)

        # un solo valor: la media
        final_prob = float(np.mean(property_preds))
        return final_prob

    def evaluate(self, round_num):
        y_true = []
        y_pred = []
        all_losses = []

        for client in self.clients:
            loss, num_samples, metrics = client.evaluate(self.global_model.state_dict())
            all_losses.append(loss)

            # extiende y_true
            y_true.extend(client.y_label.tolist())
            # pred
            with torch.no_grad():
                X_input = client.X.clone().detach()
                outputs = self.global_model(X_input)  # o client.model(X_input)
                preds = (outputs.squeeze() > 0.5).int().cpu().numpy()
                y_pred.extend(preds)

        # Y ahora comparas y_pred vs y_true
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.shape != y_pred.shape:
            logger.error(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

        # Asegúrate de no reusar 'pred_labels' local con 'y_true' global
        avg_loss = float(np.mean(all_losses))
        # Accuracy global
        avg_accuracy = (y_pred == y_true).astype(float).mean()

        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        auc = roc_auc_score(y_true, y_pred)

        logger.info(f"Round {round_num} - Loss: {avg_loss}, Acc: {avg_accuracy}, AUC: {auc}")

        return {
            'round': round_num,
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

    def select_clients(self, round_num):
        """
        Selecciona 5 clientes en cada ronda, alternando entre clientes con y sin la propiedad.
        """
        clients_with_property = [c for c in self.clients if c.data['has_property']]
        clients_without_property = [c for c in self.clients if not c.data['has_property']]

        # lóg. ejemplo: en rondas pares -> con propiedad
        if round_num % 2 == 0:
            return clients_with_property, True
        else:
            return clients_without_property, False


def normalize_updates(client_updates):
    """
    Normaliza las actualizaciones de los clientes con RobustScaler, si lo deseas.
    """
    scaler = RobustScaler()
    processed_updates = torch.stack([upd.flatten() for upd in client_updates])  # shape [N, dims]
    scaled = scaler.fit_transform(processed_updates.numpy())
    return torch.tensor(scaled)


def average_client_updates(client_updates):
    """
    Promedia las actualizaciones (en la dimensión 0).
    """
    if len(client_updates) == 0:
        raise ValueError("La lista de client_updates está vacía. No se puede calcular el promedio.")

    # normalizar primero, se puede:
    normalized = normalize_updates(client_updates)
    return normalized.mean(dim=0)

    # O si no
    #stacked = torch.stack(client_updates, dim=0)
    #return stacked.mean(dim=0)
