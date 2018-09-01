import torch
from torch.utils.data import DataLoader
import pandas as pd

from VADDataset import VADDataset
from NeuralNetwork import NeuralNetwork


def main():
    # CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Se carga el dataset
    dataset = VADDataset()

    # Se separan datos de entrenamiento y de prueba
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Se crean dataloaders para obener datos en batchs, y poder hacer shuffle de los datos
    train_dataloader = DataLoader(train_dataset, batch_size=8,
                                  shuffle=True, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=8,
                                 shuffle=True, num_workers=12)

    # Se crea la red neuronal, con el uso de CUDA
    my_neural_network = NeuralNetwork()
    my_neural_network.to(device)

    # Se usar cross entropy para la perdida y el optimizador Adam
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_neural_network.parameters(), lr=0.001)

    # Dataframes para luego plotear resultados
    df_acuracy_epoch = pd.DataFrame(columns=['epochs', 'accuracy'])
    df_error = pd.DataFrame(columns=['epochs', 'epochs_number', 'loss'])

    # Se prueba la red con distinto epochs
    test_epochs = [1, 5]
    for test_epoch in test_epochs:
        for epoch in range(test_epoch):
            running_loss = 0.0
            epoch_loss = 0.0
            for i, data_batch in enumerate(train_dataloader):
                # Se obtienen datos del datase y se pasan a la GPU
                audio, label = data_batch
                audio, label = audio.to(device), label.to(device)

                optimizer.zero_grad()

                # Se pasan los datos por la red y se hace el backpropagation
                output = my_neural_network(audio)
                output = output.to(device)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                # Se obtienen los errores
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 500 == 499:  # print every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, (i + 1) * 4, running_loss / (4 * 500)))
                    running_loss = 0.0
            df_error.loc[len(df_error)] = {'epochs': test_epoch, 'epoch_number': epoch + 1,
                                           'loss': epoch_loss / train_size}
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                audio, labels = data
                audio, labels = audio.to(device), labels.to(device)
                outputs = my_neural_network(audio)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        df_acuracy_epoch.loc[len(df_acuracy_epoch)] = {'epochs': test_epoch, 'accuracy': correct / total}
        print('Accuracy of the network on the %d test audios: %d %%' % (len(test_dataset),

                                                                        100 * correct / total))
    df_acuracy_epoch.to_csv('epochs_accuracy.csv', index=False)
    df_error.to_csv('error_epoch.csv', index=False)
    return 1


if __name__ == '__main__':
    main()
