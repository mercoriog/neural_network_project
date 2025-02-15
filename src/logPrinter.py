import sys
from datetime import datetime

# Classe che duplica l'output su console e su file
class DualOutput:
    def __init__(self, *outputs):
        self.outputs = outputs
    
    def write(self, message):
        for output in self.outputs:
            output.write(message)
            # Forza la scrittura immediata
            output.flush()
    
    def flush(self):
        for output in self.outputs:
            output.flush()

def createBasename(act_func, num_layer, num_neurons, num_epochs):
    # Ottiengo il timestamp corrente
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Creo il nome del file con il timestamp
    basename = f"log_F{act_func}_L{num_layer}_N{num_neurons}_E{num_epochs}_T{timestamp}"

    return basename    

def initLogger(filename):
    # Creo il file di log
    log_file = open(filename, 'w')

    # Salva l'output standard originale
    stdout_originale = sys.stdout

    # Reindirizzo l'output sia su console che su file
    sys.stdout = DualOutput(sys.stdout, log_file)

    return log_file, stdout_originale

def closeLogger(log_file,  stdout_originale):
    # Forza il flush per assicurarmi che tutti i dati siano scritti sul file
    sys.stdout.flush()

    # Ripristina stdout originale (console)
    sys.stdout = stdout_originale

    # Chiudo il file
    log_file.close()