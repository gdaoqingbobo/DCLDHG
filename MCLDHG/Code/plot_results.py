import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from Utils.Utils import get_metrics
import os

def plot_all_curves(results_dict, save_dir='Figures'):
    """

    """

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    

    os.makedirs(save_dir, exist_ok=True)
    

    colors = ['red', 'blue', 'purple', 'cyan', 'green', 'orange', 'pink']
    

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(20, 8))
    

    for i, (model_name, data) in enumerate(results_dict.items()):
        # 计算指标
        metrics = get_metrics(data['y_true'], data['y_pred'])
        aupr, auc = metrics[:2]
        std = data.get('std', {'auc': 0, 'aupr': 0})
        

        fpr, tpr, _ = roc_curve(data['y_true'], data['y_pred'][:, 1])
        ax_roc.plot(fpr, tpr, color=colors[i], 
                   label=f'{model_name} ({auc:.4f} ± {std["auc"]:.4f})',
                   linewidth=2)
        

        precision, recall, _ = precision_recall_curve(data['y_true'], 
                                                    data['y_pred'][:, 1])
        ax_pr.plot(recall, precision, color=colors[i],
                  label=f'{model_name} ({aupr:.4f} ± {std["aupr"]:.4f})',
                  linewidth=2)
    

    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('(A) ROC curves for LDA prediction on dataset 1')
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True)
    

    ax_pr.set_xlim([0.2, 1.0])
    ax_pr.set_ylim([0.0, 0.7])
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('(B) PR curves for LDA prediction on dataset 1')
    ax_pr.legend(loc="upper right", fontsize=8)
    ax_pr.grid(True)
    

    plt.tight_layout()
    

    fig.savefig(f'{save_dir}/roc_pr_curves.pdf', format='pdf', 
                dpi=300, bbox_inches='tight')
    plt.close()

def load_results(result_path):

    pass

def main():

    results = {
        'GCLMTP': {
            'y_true': np.load('results/GCLMTP_true.npy'),
            'y_pred': np.load('results/GCLMTP_pred.npy'),
            'std': {'auc': 0.0014, 'aupr': 0.0021}
        },
        'GCN': {
            'y_true': np.load('results/GCN_true.npy'),
            'y_pred': np.load('results/GCN_pred.npy'),
            'std': {'auc': 0.0022, 'aupr': 0.0020}
        }

    }
    

    plot_all_curves(results, save_dir='Figures')
    print(" Figures/roc_pr_curves.pdf")

if __name__ == "__main__":
    main()