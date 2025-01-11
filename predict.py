
def predict(model, X_test, y_test):
    predictions = model.predict(X_test)
    index = 0  # up to 9999
    print("Ground truth for test digit: ", y_test[index])
    print("\n")
    print("Predictions for each class:\n")
    for i in range(10):
        print("digit:", i, " probability: ", predictions[index][i])