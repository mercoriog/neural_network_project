import matplotlib as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plotResults(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(15, 4))

    # If metric_name is not a list or tuple, convert metrics and metric_name to lists
    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]

    # Plot each metric with its corresponding color
    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])

    # Set labels and title
    plt.xlabel("Epoch")  
    plt.ylabel(ylabel)  
    plt.title(title)     

    # Set axis limits
    plt.xlim([0, 20])    
    plt.ylim(ylim)       

    # Customize x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))  
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  
    ax.xaxis.set_minor_locator(MultipleLocator(1))  

    # Add grid and legend
    plt.grid(True)  
    plt.legend(metric_name)  

    # Display the plot
    plt.show()
    plt.close()  