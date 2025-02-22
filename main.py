from src import predicter
from src import datasetLoader as dsLoad
from src import trainer
from src import confusionMatrix as confMx
from src import modelCompiler as compiler
from src import logPrinter
import os

if __name__ == "__main__":
    # Richiesta dei parametri all'utente tramite input da tastiera
    input_size = 784
    num_neurons = int(input("Inserisci il numero di neuroni per strato nascosto: "))
    hidden_num_layers = int(input("Inserisci il numero di strati nascosti: "))
    func_activation = input("Inserisci la funzione di attivazione (relu/leaky_relu/tanh): ")
    num_classes = 10
    epochs = int(input("Inserisci il numero di epoche di addestramento: "))

    # Crea il basename per salvare i log
    basename = logPrinter.createBasename(func_activation, hidden_num_layers, num_neurons, epochs)

    # Creo il nome del percorso di salvataggio
    save_path = os.path.join("out", basename)

    # Creo la cartella
    os.makedirs(save_path, exist_ok=True)

    # Creo il filename per i log testuali
    filename_txt = os.path.join(save_path, f"log_{basename}.txt")

    # Creo il filename per il grafico loss
    filename_graph_loss = os.path.join(save_path, f"Loss_{basename}.png")

    # Creo il filename per il grafico accuracy
    filename_graph_accuracy = os.path.join(save_path, f"Accuracy_{basename}.png")

    # Creo il filename per il grafico matrix
    filename_graph_matrix = os.path.join(save_path, f"Matrix_{basename}.png")

    # Inizializzo il logPrinter
    log_file, stdout_originale = logPrinter.initLogger(filename_txt)

    # Stampo la configurazione iniziale
    print(f"Activation Function: {func_activation}\nHidden Layers: {hidden_num_layers}\nInternal Neurons: {num_neurons}\nEpochs: {epochs}")

    # Carica i DataLoader
    train_loader, valid_loader, test_loader = dsLoad.getData()

    # Definisci il modello
    model = compiler.NN(input_size, num_neurons, hidden_num_layers, func_activation, num_classes)

    # Stampo il modello
    print(model)

    # Addestra il modello
    training_results = trainer.startTraining(model, train_loader, valid_loader, epochs)

    # Mostra i risultati dell'addestramento
    trainer.showTrainingResults(training_results, epochs, filename_graph_loss, filename_graph_accuracy)

    # Effettua le previsioni sul test set
    predictions, labels, accuracies = predicter.predict(model, test_loader)

    # Visualizza la matrice di confusione
    metrics = confMx.confusionMatrix(model, test_loader, filename_graph_matrix)

    print(metrics)

    print('Done.')

    logPrinter.closeLogger(log_file, stdout_originale)