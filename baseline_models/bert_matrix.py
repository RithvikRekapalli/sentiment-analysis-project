import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# BERT Confusion Matrix
confusion_matrix_bert = np.array([[1860, 0], [140, 0]])

# Plotting the confusion matrix for BERT
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_bert, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix for BERT')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("BERT Confusion Matrix:")
print(confusion_matrix_bert)