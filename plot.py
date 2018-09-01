import pandas as pd
import matplotlib.pyplot as plt



def accuracy_vs_epoch():
    accuracy = pd.read_csv('epochs_vs_accuracy.csv')
    plt.plot(accuracy['epochs'], accuracy['accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Precisión")
    plt.title("Epochs v/s Precisión")
    plt.savefig('epochs_vs_accuracy.png')
    plt.show()

def error_epoch_20():
    df = pd.read_csv('error_vs_epoch.csv')
    df = df[df['epochs'] == 20]
    plt.plot(df['epochs_number'], df['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Perdida")
    plt.title("Epochs v/s Loss")
    plt.savefig('epochs_vs_loss.png')
    plt.show()

error_epoch_20()