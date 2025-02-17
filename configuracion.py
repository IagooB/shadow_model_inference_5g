PREFIJO_SAVE = "resultados_iniciales"

SEED = 42


def load_config():
    """
    Carga los parámetros de configuración en un diccionario para facilitar su uso.
    """
    return {
        "num_rounds": 30,
        "num_clients": 10,
        "random_seed": 42,
        "batch_size": 32,

        "lr_cliente_simulado": 0.001,

        "num_shadow_models": 50, # Tiene que ser par
        "global_model_epochs": 10,
        "shadow_train_rounds": 10,
        "shadow_data_fraction": 0.01,
        "lr_shadow_training": 0.001,
        "batch_size_attack_model": 8,

        "prob_range": 0.1,
        "clients_random": False,
        "fraction": 0.2,

        "aleatoriedad": False,

        # ATAQUE #
        "epochs_ataque": 50,
        "lr_ataque": 0.0001,

        # RUIDO #
        "aplicar_ruido": False,
        "ruido_obj": ["gradients"],
        "ruido_per": 0.4,  # Proporción de datos afectados por ruido
        "noise_std": 0.2,   # Desviación estándar del ruido
        "epsilon": 1.0,     # Parámetro de privacidad diferencial
        "delta": 1e-5,      # Delta para ruido gaussiano
        "sensitivity": 1.0, # Sensibilidad del mecanismo de ruido
        "privacy_type": "gaussian",  # Tipo de ruido a aplicar
        "selected_layers": "all", # [0, -1]  # Aplica ruido a todas las capas

        # LABEL FLIPPING #
        "label_flipping": False,
        "flipping_antes": False,
        "prob_flip_0": 0.2,
        "prob_flip_1": 0.2,
        "flip_target": "Slice",

        # RESULTADOS #
        "rondas_a_ignorar": 10,
        "property_threshold": 0.5,
        "learning_rate": 0.1,
        "data_file_path": '../label_bi_10.csv',
        "prefijo_save": "resultados_iniciales",
    }

exper_config = load_config()