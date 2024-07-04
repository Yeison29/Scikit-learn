import torch as t
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class Modelo(t.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = t.nn.Linear(1, 3)
        self.linear2 = t.nn.Linear(3, 7)
        self.linear3 = t.nn.Linear(7, 3)
        self.linear4 = t.nn.Linear(3, 1)

        self.relu = t.nn.ReLU()
        self.softmax = t.nn.Softmax()

        self.loss_function = t.nn.MSELoss()
        self.optimizer = t.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, state):
        x = t.tensor(state)
        x = self.relu(self.linear1(x))
        x = self.softmax(self.linear2(x))
        return t.sigmoid(x)

    def training_loop(self, train_loader, modelo):
        self.train()
        epochs = 50
        for epoch in range(epochs):

            loss_tracker = []

            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = modelo(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_tracker.append(loss.item())

            if epoch % 1 == 0:
                epoch_loss = sum(loss_tracker) / len(loss_tracker)
                print(f"Epoca {epoch + 1}/{epochs}. Loss: {epoch_loss}")

    def test(self, y_input, y_output):
        self.eval()
        y_pred = self.forward(y_input)
        loss = self.loss_function(y_pred, y_output)
        return loss


def main():
    modelo = Modelo()
    fahrenheit = np.linspace(-450, 500, 100000)

    celsius = [(x - 32) * 5 / 9 for x in fahrenheit]

    fahrenheit = fahrenheit.reshape(-1, 1)

    celsius = np.array(celsius).reshape(-1, 1)

    batch_size = 64

    train_data = TensorDataset(t.Tensor(fahrenheit), t.Tensor(celsius))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    modelo.training_loop(train_loader, modelo)

    fahrenheit = t.tensor([77.0])
    celsius_calculation = (fahrenheit - 32) * 5 / 9

    celsius_prediction = modelo.test(fahrenheit, celsius_calculation)

    print('Prediccion: ', celsius_prediction)


if __name__ == '__main__':
    main()
