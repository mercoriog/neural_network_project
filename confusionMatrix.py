import tensorflow as tf
import numpy as np
import matplotlib as plt
import seaborn as sn

def confusionMatrix(model, X_test, y_test):
    # Generate predictions for the test dataset.
    predictions = model.predict(X_test)

    # For each sample image in the test dataset, select the class label with the highest probability.
    predicted_labels = [np.argmax(i) for i in predictions]

    # Convert one-hot encoded labels to integers.
    y_test_integer_labels = tf.argmax(y_test, axis=1)

    # Generate a confusion matrix for the test dataset.
    cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

    # Plot the confusion matrix as a heatmap.
    plt.figure(figsize=[15, 8])
    

    sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 14})
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()