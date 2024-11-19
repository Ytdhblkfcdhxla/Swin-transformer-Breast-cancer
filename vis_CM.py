import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors


with open('results/', 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)

    pred_list0 = []
    true_list0 = []
    pred_list1 = []
    true_list1 = []
    pred_list2 = []
    true_list2 = []

    for r in csv_reader:
        try:

            pred_list0.append(float(r[1]))
            pred_list1.append(float(r[2]))
            pred_list2.append(float(r[3]))


            if int(r[-1]) == 0:
                true_list0.append(1)
                true_list1.append(0)
                true_list2.append(0)
            elif int(r[-1]) == 1:
                true_list0.append(0)
                true_list1.append(1)
                true_list2.append(0)
            elif int(r[-1]) == 2:
                true_list0.append(0)
                true_list1.append(0)
                true_list2.append(1)

        except ValueError as e:
            print(f"Skipping row with error: {e}, data: {r}")


y_pred_combined = np.array([pred_list0, pred_list1, pred_list2]).T.argmax(axis=1)
y_true_combined = np.array([true_list0, true_list1, true_list2]).T.argmax(axis=1)


cm = confusion_matrix(y_true_combined, y_pred_combined)


cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


orange_cmap = mcolors.LinearSegmentedColormap.from_list('orange_cmap', [(1, 1, 0.8), (1, 0.6, 0)], N=256)


fig, ax = plt.subplots(figsize=(10, 10))

def plot_with_percentage(ax, cm_normalized, cmap):
    cax = ax.matshow(cm_normalized, cmap=cmap)

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            value = cm_normalized[i, j]
            text = f'{value*100:.2f}%' if value > 0 else ''
            ax.text(j, i, text, ha='center', va='center', color='black', fontsize=20)  # 调整字体大小
    return cax


cax = plot_with_percentage(ax, cm_normalized, orange_cmap)


class_names = ['HER2-zero', 'HER2-low', 'HER2-positive']
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, fontsize=18)
ax.set_yticklabels(class_names, fontsize=18)

ax.set_xlabel('Predicted labels', fontsize=18)
ax.set_ylabel('True labels', fontsize=18)


ax.tick_params(axis='both', which='major', labelsize=20)

plt.title('Normalized Confusion Matrix', fontsize=20)  # 调整标题字体大小


cbar = plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Percentage')


plt.tight_layout()


plt.savefig('Figures/confusion_matrix/', format='tiff', dpi=300)
plt.show()
