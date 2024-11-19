import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Helper function to compute the midrank, kept from the original implementation
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2

# Load the data from CSV
with open('results/', 'r') as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)  # 跳过表头

    pred_list0 = []
    true_list0 = []
    pred_list1 = []
    true_list1 = []
    pred_list2 = []
    true_list2 = []

    for r in csv_reader:
        try:
            # 读取并转换预测值
            pred_list0.append(float(r[1]))  # score_0
            pred_list1.append(float(r[2]))  # score_1
            pred_list2.append(float(r[3]))  # score_2

            # 读取真实标签，使用 'gt' 列的值决定属于哪个类别
            if int(r[5]) == 0:  # 如果 'gt' 为 0
                true_list0.append(1)
                true_list1.append(0)
                true_list2.append(0)
            elif int(r[5]) == 1:  # 如果 'gt' 为 1
                true_list0.append(0)
                true_list1.append(1)
                true_list2.append(0)
            elif int(r[5]) == 2:  # 如果 'gt' 为 2
                true_list0.append(0)
                true_list1.append(0)
                true_list2.append(1)

        except ValueError as e:
            print(f"Skipping row with error: {e}, data: {r}")

pred_list = [pred_list0, pred_list1, pred_list2]
true_list = [true_list0, true_list1, true_list2]

# Function to calculate various metrics for each class
def calculate_metrics(y_true, y_pred):
    y_pred_binary = [1 if y >= 0.5 else 0 for y in y_pred]

    # AUC
    roc_auc = roc_auc_score(y_true, y_pred)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred_binary)

    # Precision
    precision = precision_score(y_true, y_pred_binary)

    # Sensitivity (Recall)
    sensitivity = recall_score(y_true, y_pred_binary)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    specificity = tn / (tn + fp)

    # F1 Score
    f1 = f1_score(y_true, y_pred_binary)

    return roc_auc, accuracy, precision, sensitivity, specificity, f1

# Bootstrap function to calculate the 95% confidence interval for AUC
def bootstrap_auc(y_true, y_pred, n_bootstrap=1000, alpha=0.95):
    np.random.seed(42)  # 固定随机种子，确保一致性
    aucs = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        aucs.append(roc_auc_score(y_true_boot, y_pred_boot))

    lower_bound = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(aucs, (1 + alpha) / 2 * 100)

    return np.mean(aucs), lower_bound, upper_bound

# Calculate metrics and save to CSV
metrics_data = []
class_names = ['HER2-zero', 'HER2-low', 'HER2-positive']

for i in range(3):
    y_true = np.array(true_list[i])
    y_pred = np.array(pred_list[i])

    roc_auc, accuracy, precision, sensitivity, specificity, f1 = calculate_metrics(y_true, y_pred)

    # Print metrics to console
    print(f'Class: {class_names[i]}')
    print(f'AUC: {roc_auc:.3f}')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Sensitivity: {sensitivity:.3f}')
    print(f'Specificity: {specificity:.3f}')
    print(f'F1 Score: {f1:.3f}')
    print('-' * 30)

    # Append metrics to data list
    metrics_data.append({
        'Class': class_names[i],
        'AUC': roc_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1
    })

# Write the metrics to a CSV file in the specified directory
output_file_path = 'Figures/'
with open(output_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Class', 'AUC', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for data in metrics_data:
        writer.writerow(data)

print(f"Metrics saved to {output_file_path}")

# Function to plot combined ROC curve using roc_auc_score for consistency
def plot_combined_roc_curve_with_roc_auc(pred_list, true_list):
    plt.figure(figsize=(8, 6))


    class_names = ['HER2-zero', 'HER2-low', 'HER2-positive']
    colors = ['darkorange', 'blue', 'green']

    for i in range(3):
        y_pred = np.array(pred_list[i])
        y_true = np.array(true_list[i])


        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)


        mean_auc, ci_lower, ci_upper = bootstrap_auc(y_true, y_pred)


        print(f'{class_names[i]} AUC: {mean_auc:.3f}')
        print(f'{class_names[i]} 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')


        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{class_names[i]} AUC = {mean_auc:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")

    plt.savefig('Figures/ROC/FY_combined_roc_with_roc_auc.tif', format='tiff', dpi=300)
    plt.show()


plot_combined_roc_curve_with_roc_auc(pred_list, true_list)
