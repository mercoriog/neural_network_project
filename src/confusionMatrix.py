import torch
import numpy as np
import matplotlib as plt
import seaborn as sn

def confusionMatrix(model, X_test, y_test):

    # This function sets the model to evaluation mode. 
    # This mode is crucial for ensuring consistent model behavior during prediction/inference.
    model.eval()

    # PyTorch models only accept input in the torch.Tensor format. 
    # If your data is in another format, it must be converted.
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.long)

    # During validation, gradients aren't needed because the model isn't being updated. 
    # This also reduces memory consumption and increases speed.
    with torch.no_grad():
        # Generate predictions for the test dataset.
        predictions = model(X_test)

    # Determine the predicted class with the highest probability.
    predicted_labels = torch.argmax(predictions, dim=1)

    # Create the confusion matrix.
    cm = torch.zeros(len(torch.unique(y_test)), len(torch.unique(y_test)), dtype=torch.int32)
    for true_label, pred_label in zip(y_test, predicted_labels):
        cm[true_label, pred_label] += 1

    # Convert the confusion matrix to a NumPy array.
    cm = cm.numpy()

    # View Confusion Matrix.
    plt.figure(figsize=(15, 8))
    sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 14})
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()