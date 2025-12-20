import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred) 
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "F1-Score": f1
    }

def save_confusion_matrix(y_true, y_pred, model_name, folder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen')
    plt.savefig(f'outputs/{folder}/cm_{model_name}.png')
    plt.close()

def save_roc_curve(y_true, y_probs, model_name, folder):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'outputs/{folder}/roc_{model_name}.png')
    plt.close()

def save_ysa_plot(history, folder):
    plt.figure(figsize=(12, 4))
    
    # Accuracy Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Doğrulama')
    plt.title('YSA Doğruluk (Accuracy)')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Doğrulama')
    plt.title('YSA Kayıp (Loss)')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig(f'outputs/{folder}/ysa_performance.png')
    plt.close()