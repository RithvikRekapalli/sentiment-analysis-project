import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Roberta Evaluation Metrics from checkpoint-6000
accuracy = 0.9526
precision = 0.9396
recall = 0.9673
total_samples = 25000  # Assuming the evaluation was done on the test set
num_classes = 2
balanced_samples_per_class = total_samples / num_classes

# Calculate True Positives (TP) - assuming recall is for the positive class (label 1)
tp = int(recall * balanced_samples_per_class)

# Calculate False Positives (FP)
fp = int((tp / precision) - tp)
fp = max(0, fp) # Ensure FP is not negative

# Calculate False Negatives (FN)
fn = int(balanced_samples_per_class - tp)
fn = max(0, fn) # Ensure FN is not negative

# Calculate True Negatives (TN)
tn = int(total_samples - tp - fp - fn)
tn = max(0, tn) # Ensure TN is not negative

# Create the confusion matrix
confusion_matrix_roberta = np.array([[tn, fp], [fn, tp]])

# Plotting the confusion matrix for Roberta
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_roberta, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Estimated Confusion Matrix for Roberta (Checkpoint 6000)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Estimated Roberta Confusion Matrix (Checkpoint 6000):")
print(confusion_matrix_roberta)