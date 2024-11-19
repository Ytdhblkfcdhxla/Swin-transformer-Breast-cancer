import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


with open('results/', 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)


    y_true = []
    y_scores = []


    for r in csv_reader:
        try:

            y_true.append(int(r[-1]))
            y_scores.append([float(r[1]), float(r[2]), float(r[3])])
        except ValueError as e:
            print(f"Skipping row with error: {e}, data: {r}")


y_true = np.array(y_true)
y_scores = np.array(y_scores)


y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])


plt.figure(figsize=(10, 8))
n_classes = y_true_binarized.shape[1]


class_names = ['HER2-zero', 'HER2-low', 'HER2-positive']

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_scores[:, i])
    average_precision = average_precision_score(y_true_binarized[:, i], y_scores[:, i])
    plt.plot(recall, precision, marker='o', label=f'{class_names[i]} (AP = {average_precision:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for HER2 Classes')
plt.legend(loc='best')


plt.savefig('Figures/P-R/', format='tiff', dpi=300)
plt.show()
