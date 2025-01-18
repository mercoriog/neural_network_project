import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plotResults(metrics, reps, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    """
    Visualizza una o più metriche in un grafico.

    Args:
        metrics (list o array): Una lista di metriche da visualizzare.
        title (str, opzionale): Titolo del grafico.
        ylabel (str, opzionale): Etichetta dell'asse y.
        ylim (list, opzionale): Limiti dell'asse y (es. [min, max]).
        metric_name (list, opzionale): Nomi delle metriche per la legenda.
        color (list, opzionale): Colori per ciascuna metrica.
    """
    # Crea una figura e un asse per il grafico
    fig, ax = plt.subplots(figsize=(15, 4))

    # Se metric_name non è una lista o una tupla, converti metrics e metric_name in liste
    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics]
        metric_name = [metric_name]

    # Se color non è specificato, usa colori predefiniti
    if color is None:
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Colori predefiniti

    # Plotta ciascuna metrica con il colore corrispondente
    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx], label=metric_name[idx])

    # Imposta titolo ed etichette degli assi
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel("Epoch")

    # Imposta i limiti degli assi
    if ylim:
        plt.ylim(ylim)
    plt.xlim([0, reps])  # Limite fisso per l'asse x (da 0 a N epoch)

    # Personalizza i segni sull'asse x
    ax.xaxis.set_major_locator(MultipleLocator(5))  # Segni principali ogni 5 reps
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # Formato intero
    ax.xaxis.set_minor_locator(MultipleLocator(1))  # Segni minori ogni 1 reps

    # Aggiungi griglia e legenda
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Griglia principale e secondaria
    if metric_name:
        plt.legend()

    # Mostra il grafico
    plt.show()
    plt.close()