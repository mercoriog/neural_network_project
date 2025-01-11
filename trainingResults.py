from plotsResults import plotResults

def trainingResults(training_results):
    # Retrieve training results.
    train_loss = training_results.history["loss"]
    train_acc  = training_results.history["accuracy"]
    valid_loss = training_results.history["val_loss"]
    valid_acc  = training_results.history["val_accuracy"]

    plotResults(
        [train_loss, valid_loss],
        ylabel="Loss",
        ylim=[0.0, 0.5],
        metric_name=["Training Loss", "Validation Loss"],
        color=["g", "b"],
    )

    plotResults(
        [train_acc, valid_acc],
        ylabel="Accuracy",
        ylim=[0.9, 1.0],
        metric_name=["Training Accuracy", "Validation Accuracy"],
        color=["g", "b"],
    )