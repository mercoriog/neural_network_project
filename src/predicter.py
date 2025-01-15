import torch

def predict(model, X_test, y_test):
    # Imposta il modello in modalità di valutazione
    model.eval()
    
    # Disabilita il calcolo dei gradienti per l'inferenza
    with torch.no_grad():
        # Effettua le previsioni
        predictions = model(X_test)
    
    # Seleziona un indice per visualizzare i risultati
    index = 0  # fino a 9999
    
    # Stampa la vera etichetta (ground truth) per il dato di test
    print("Ground truth for test digit: ", y_test[index].item())
    print("\n")
    
    # Stampa le probabilità previste per ogni classe
    print("Predictions for each class:\n")
    for i in range(10):
        print("digit:", i, " probability: ", predictions[index][i].item())