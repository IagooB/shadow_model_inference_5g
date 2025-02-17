from client import *
from data import *
from salida import *
from server import *
import random

#### EXPERIMENT ####

def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed()

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"


def validate_configuration():
    """
    Valida las configuraciones de parámetros y asegura que las combinaciones sean coherentes.
    """
    logger.info("Validating configuration...")

    # Informar configuraciones redundantes
    if not exper_config["label_flipping"] and exper_config["flipping_antes"]:
        raise ValueError("FLIPPING_ANTES=True no tiene sentido cuando LABEL_FLIPPING=False.")

    if not exper_config["aplicar_ruido"] and not exper_config["label_flipping"]:
        logger.info(
            "Ni ruido ni label flipping están activados. El experimento no incluye perturbaciones en los datos.")

    # Validar parámetros relacionados con ruido
    if exper_config["aplicar_ruido"]:
        if exper_config["epsilon"] <= 0:
            raise ValueError("EPSILON debe ser mayor que 0.")
        if exper_config["delta"] is not None and not (0 < exper_config["delta"] < 1):
            raise ValueError("DELTA debe estar entre 0 y 1 si se usa ruido gaussiano.")
        if exper_config["sensitivity"] <= 0:
            raise ValueError("SENSITIVITY debe ser mayor que 0.")
        if not all(obj in ["gradients", "data"] for obj in exper_config["ruido_obj"]):
            raise ValueError("RUÍDO_OBJ solo puede contener 'gradients' o 'data'.")

    # Validar parámetros relacionados con flipping
    if exper_config["label_flipping"]:
        if not (0 <= exper_config["prob_flip_0"] <= 1):
            raise ValueError("PROB_FLIP_0 debe estar entre 0 y 1.")
        if not (0 <= exper_config["prob_flip_1"] <= 1):
            raise ValueError("PROB_FLIP_1 debe estar entre 0 y 1.")

    logger.info("Configuration validation completed successfully.")

def main():
    set_seed(42)
    validate_configuration()

    # 1) Cargar data
    df = load_data(exper_config['data_file_path'])
    X, y_label, y_slice = preprocess_data(df, exper_config['fraction'])

    # 2) Dividir en global y shadow
    (X_global, y_label_global, y_slice_global), \
    (X_shadow, y_label_shadow, y_slice_shadow), _ = split_data_for_models(
        X, y_label, y_slice
    )

    # 3) Crear datos de clientes
    client_data = create_client_data(X_global, y_label_global, y_slice_global)

    # 4) Modelo global
    global_model = create_global_model(client_data[0]['X'].shape[1])

    # 5) Inicializar clientes
    clients = [SimulatedFlowerClient(i, data, global_model, exper_config, batch_size=exper_config["batch_size"])
               for i, data in enumerate(client_data)]

    # 6) Shadow training → produce (shadow_models, X_attack, y_attack)
    from models import train_shadow_models
    shadow_models, X_attack, y_attack = train_shadow_models(
        X_shadow=torch.tensor(X_shadow, dtype=torch.float32),
        y_label_shadow=torch.tensor(y_label_shadow, dtype=torch.float32),
        y_slice_shadow=torch.tensor(y_slice_shadow, dtype=torch.float32),
        global_model=global_model
    )

    # 7) Entrenamos Attack Model (ejemplo RandomForest)

    X_attack_np = np.stack(X_attack, axis=0)
    y_attack_np = np.array(y_attack)

    input_dim = X_attack_np.shape[1]
    attack_model = AttackModelNN(input_dim)

    # Entrenamos
    attack_model.train_model(
        X_train=X_attack_np,
        y_train=y_attack_np,
        epochs=exper_config["epochs_ataque"],
        lr=exper_config["lr_ataque"],
        batch_size=exper_config["batch_size_attack_model"]
    )
    logger.info("Attack Model NN entrenado con updates etiquetados en PyTorch.")

    # 8) Creamos el servidor pasando attack_model
    from server import FederatedServer
    federated_server = FederatedServer(global_model, clients, attack_model, exper_config['num_rounds'])

    # 9) Ejecución FL
    results = federated_server.train()

    # 10) Salidas, métricas, etc.
    transitions = analyze_probability_transitions(results)
    metrics, fpr, tpr, thresholds = calculate_and_log_metrics(results)
    save_results_to_csv(results, transitions, metrics)
    plot_results(results, exper_config["prefijo_save"])
    plot_combined_roc_threshold(
        fpr, tpr, thresholds,
        metrics['ROC AUC'], metrics['Optimal Threshold'],
        exper_config["prefijo_save"]
    )

    logger.info("Federated learning simulation completed")


if __name__ == "__main__":
    main()
