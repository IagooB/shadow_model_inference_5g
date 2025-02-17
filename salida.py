import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap, entropy
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, precision_recall_curve, accuracy_score, \
    precision_score, recall_score

from configuracion import *
from logg import *

#### OUTPUTS ####

# Rondas a ignorar
ronda_ignorar = int(exper_config["rondas_a_ignorar"])


def calculate_correct_predictions(results):
    correct_predictions = 0
    incorrect_predictions = 0
    dynamic_thresholds = calculate_dynamic_threshold(results)

    for i in range(3, len(results)):
        current_prob = results[i]['property_probability']
        previous_prob = results[i - 1]['property_probability']
        previous_property = results[i - 1]['has_property']
        current_property = results[i]['has_property']
        threshold = dynamic_thresholds[i]

        if not previous_property and not current_property:
            is_correct = abs(current_prob - previous_prob) < threshold * 0.1
        elif previous_property and not current_property:
            is_correct = current_prob < previous_prob
        elif not previous_property and current_property:
            is_correct = current_prob > previous_prob
        elif previous_property and current_property:
            is_correct = abs(current_prob - previous_prob) < threshold * 0.1
        else:
            is_correct = False

        if is_correct:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    total_predictions = correct_predictions + incorrect_predictions
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    logger.info(f"Correct predictions: {correct_predictions}, "
                f"Incorrect predictions: {incorrect_predictions}, Accuracy: {accuracy}")
    return correct_predictions, incorrect_predictions, dynamic_thresholds


def plot_results(results, output_path, show=False):
    """
    Genera gráficos de la evolución de la probabilidad detectada, pérdida y precisión.

    Parameters:
        results (list): Resultados de la simulación.
        output_path (str): Directorio para guardar las imágenes.
        show (bool): Si es True, muestra la gráfica al finalizar la ejecución.
    """
    # Ignoramos rondas

    round_nums = [r['round'] for r in results][ronda_ignorar:]
    probabilities = [r['property_probability'] for r in results][ronda_ignorar:]
    thresholds = [r['threshold_used'] for r in results][ronda_ignorar:]
    losses = [r['loss'] for r in results][ronda_ignorar:]
    accuracies = [r['accuracy'] for r in results][ronda_ignorar:]
    property_present = [r['has_property'] for r in results][ronda_ignorar:]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Gráfico 1: Probabilidad por ronda
    axs[0].plot(round_nums, probabilities, marker='o', color='blue', label='Property Probability')

    # Agregar puntos en rojo para las rondas donde la propiedad está presente
    has_property_rounds = np.array(round_nums)[property_present]
    has_property_probs = np.array(probabilities)[property_present]
    axs[0].scatter(has_property_rounds, has_property_probs, color='red', zorder=5, label='Rounds with Property')

    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Property Probability')
    axs[0].set_title('Property Probability by Round (ignoring first 3 rounds)')
    axs[0].grid(True)
    axs[0].legend()

    # Gráfico 2: Pérdida y Precisión del modelo global
    ax1 = axs[1]
    ax1.plot(round_nums, losses, label='Loss', color='red', marker='x')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(round_nums, accuracies, label='Accuracy', color='blue', marker='o')
    ax2.set_ylabel('Accuracy', color='blue')

    plt.savefig(f"{output_path}/property_probability_loss.png")


def plot_combined_roc_threshold(fpr, tpr, thresholds, auc_roc, optimal_threshold, output_path):
    """
    Genera una imagen con dos gráficos:
    - Izquierda: Curva ROC.
    - Derecha: TPR/FPR vs Threshold con umbral óptimo.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Curva ROC
    axs[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_roc:.2f})')
    axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curve')
    axs[0].legend(loc="lower right")
    axs[0].grid(True)

    # Gráfico 2: TPR/FPR vs Threshold
    axs[1].plot(thresholds, tpr, label='True Positive Rate (TPR)', color='green', lw=2)
    axs[1].plot(thresholds, fpr, label='False Positive Rate (FPR)', color='red', lw=2)
    axs[1].axvline(optimal_threshold, color='blue', linestyle='--',
                   label=f'Optimal Threshold ({optimal_threshold:.2f})')
    axs[1].set_xlabel('Threshold')
    axs[1].set_ylabel('Rate')
    axs[1].set_title('TPR and FPR vs. Threshold')
    axs[1].legend(loc="best")
    axs[1].grid(True)

    fig.tight_layout()
    plt.savefig(f"{output_path}/roc_threshold.png")
    plt.close()


def calculate_and_log_metrics(results):
    y_true = [r['has_property'] for r in results][ronda_ignorar:]
    y_prob = [r['property_probability'] for r in results][ronda_ignorar:]
    dynamic_thresholds = [r.get('threshold_used', 0.5) for r in results][ronda_ignorar:]

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    auc_roc = auc(fpr, tpr)
    auc_pr = auc(recall, precision)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5

    y_pred_dynamic = [1 if prob > thresh else 0 for prob, thresh in zip(y_prob, dynamic_thresholds)]
    y_pred_optimal = [1 if prob > optimal_threshold else 0 for prob in y_prob]

    cm = confusion_matrix(y_true, y_pred_dynamic)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Fix: Add zero_division parameter to precision_score
    precision_dyn = precision_score(y_true, y_pred_dynamic, zero_division=0)
    recall_dyn = recall_score(y_true, y_pred_dynamic)
    f1_dyn = f1_score(y_true, y_pred_dynamic)

    accuracy = accuracy_score(y_true, y_pred_optimal)

    # Fix: Add zero_division parameter to precision_score
    precision_opt = precision_score(y_true, y_pred_optimal, zero_division=0)
    recall_opt = recall_score(y_true, y_pred_optimal)
    f1_optimal = f1_score(y_true, y_pred_optimal)
    f1_ci_lower, f1_ci_upper = calculate_f1_ci(y_true, y_pred_optimal)
    entropy_val = calculate_entropy(y_prob)

    correct_transitions, incorrect_transitions, _ = calculate_correct_predictions(results)
    total_transitions = correct_transitions + incorrect_transitions
    custom_precision = (correct_transitions / total_transitions) * 100 if total_transitions > 0 else 0

    metrics = {
        'ROC AUC': auc_roc,
        'PR AUC': auc_pr,
        'Optimal Threshold': optimal_threshold,
        'F1-Score (Optimal)': f1_optimal,
        'F1-Score CI (Optimal)': f"{f1_ci_lower} - {f1_ci_upper}",
        'Accuracy': accuracy,
        'Precision (Dynamic)': precision_dyn,
        'Recall (Dynamic)': recall_dyn,
        'F1-Score (Dynamic)': f1_dyn,
        'Precision (Optimal)': precision_opt,
        'Recall (Optimal)': recall_opt,
        'Entropy': entropy_val,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn,
        'Custom Precision': custom_precision
    }

    return metrics, fpr, tpr, thresholds


def save_results_to_csv(results, transitions, metrics):
    """
    Guarda los resultados en tres CSV:
    - Resultados detallados por ronda.
    - Transiciones entre rondas.
    - Resultados finales (métricas globales).
    """
    if results:
        fieldnames = list(results[0].keys())  # Asegurar que las claves coincidan
    else:
        fieldnames = ['Round', 'Prediction', 'Probability', 'Clients with Property',
                      'Clients without Property', 'Has Property', 'Loss', 'Accuracy']

    # Resultados por Ronda
    with open(f"{PREFIJO_SAVE}/round_results.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Transiciones
    if transitions:
        fieldnames_transitions = list(transitions[0].keys())
        with open(f"{PREFIJO_SAVE}/transitions.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_transitions)
            writer.writeheader()
            for t in transitions:
                writer.writerow(t)

    # Resultados Finales
    with open(f"{PREFIJO_SAVE}/final_metrics.csv", 'w', newline='') as csvfile:
        fieldnames_metrics = list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_metrics)
        writer.writeheader()
        writer.writerow(metrics)


def calculate_dynamic_threshold(results, min_adjustment=exper_config["prob_range"]):
    """
    Calcula un threshold dinámico basado en la evolución de la probabilidad detectada.

    - Si la predicción es incorrecta, ajusta el threshold para mejorar la precisión.
    - Usa un ajuste mínimo `min_adjustment` para evitar cambios bruscos.

    Parameters:
        results (list): Resultados de la simulación.
        min_adjustment (float): Valor mínimo de ajuste del threshold.

    Returns:
        list: Lista de thresholds dinámicos por ronda.
    """
    thresholds = [exper_config["property_threshold"]]

    for i in range(1, len(results)):
        if not isinstance(i, int):
            raise TypeError(f"Expected integer index, got {type(i)} instead.")

        if not isinstance(results, (list, np.ndarray)):
            raise TypeError(f"Expected list or array for results, got {type(results)} instead.")

        prev_prob = results[i - 1]['property_probability']
        curr_prob = results[i]['property_probability']
        actual_property = results[i]['has_property']
        pred_property = results[i]['prediction']

        # Ajuste basado en errores
        if pred_property != actual_property:
            if pred_property:  # Falso positivo, threshold debe subir
                new_threshold = min(thresholds[-1] + min_adjustment, 1.0)
            else:  # Falso negativo, threshold debe bajar
                new_threshold = max(thresholds[-1] - min_adjustment, 0.0)
        else:
            # Ajustar suavemente basado en la evolución de la probabilidad
            if abs(curr_prob - prev_prob) > min_adjustment:
                new_threshold = (curr_prob + prev_prob) / 2
            else:
                new_threshold = thresholds[-1]

        thresholds.append(new_threshold)

    return thresholds


def plot_noisy_vs_clean_accuracy(noisy_results, clean_results, output_path):
    """
    Compara la precisión de FL con y sin ruido en un solo gráfico.
    """
    rounds = range(len(noisy_results))
    noisy_acc = [r['accuracy'] for r in noisy_results]
    clean_acc = [r['accuracy'] for r in clean_results]

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, noisy_acc, label='Con Ruido', color='red')
    plt.plot(rounds, clean_acc, label='Sin Ruido', color='blue')
    plt.xlabel("Rondas")
    plt.ylabel("Precisión")
    plt.title("Impacto del Ruido en la Precisión de FL")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/noisy_vs_clean_accuracy.png")
    plt.close()


def analyze_probability_transitions(results):
    """
    Analiza las transiciones de probabilidad entre rondas usando filtrado dinámico.

    Parameters:
        results (list): Resultados de la simulación.

    Returns:
        list: Lista de transiciones clasificadas.
    """
    transitions = []
    probabilities = [r['property_probability'] for r in results][ronda_ignorar:]
    rounds = [r['round'] for r in results][ronda_ignorar:]

    # Filtrar valores extremos
    prob_mean = np.mean(probabilities)
    prob_std = np.std(probabilities)
    lower_bound = prob_mean - 2 * prob_std
    upper_bound = prob_mean + 2 * prob_std
    filtered_probs = [p if lower_bound <= p <= upper_bound else prob_mean for p in probabilities]

    min_prob, max_prob = min(filtered_probs), max(filtered_probs)
    range_value = max_prob - min_prob if max_prob > min_prob else 1
    significant_move = range_value * exper_config["prob_range"]

    for i in range(1, len(filtered_probs)):
        prev_prob = filtered_probs[i - 1]
        curr_prob = filtered_probs[i]
        diff = curr_prob - prev_prob
        transition = "stable"

        if abs(diff) > significant_move:
            transition = "increase" if diff > 0 else "decrease"

        transitions.append({
            'round': rounds[i],
            'prev_probability': prev_prob,
            'current_probability': curr_prob,
            'transition': transition
        })

    logger.info(f"Transitions analyzed: {len(transitions)} processed.")
    return transitions


def calculate_f1_ci(y_true, y_pred, confidence_level=0.95, n_resamples=1000):
    f1_scores = [f1_score(y_true, np.random.permutation(y_pred)) for _ in range(n_resamples)]
    ci_lower, ci_upper = np.percentile(f1_scores, [(1 - confidence_level) * 50, (1 + confidence_level) * 50])
    return ci_lower, ci_upper


def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)


def calculate_entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
    return entropy(probabilities)

