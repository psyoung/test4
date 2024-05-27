### CS106EA Exploring Artificial Intelligence
#   Patrick Young, Stanford University
#
#   This code is mean to be loaded by Colab Notebook
#   It's purpose is to hide most of the implementation code
#   making the Colab Notebook easier to follow.

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# import numpy as np
import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML, Button, Output

from IPython.display import clear_output

from typing import Dict

import math
import random
import copy

import time

MIN_X = 0.0
MAX_X = 1.2

MIN_Y = -0.2
MAX_Y = 1.2

# the actual function that we're going to try to get our
# neural networks to simulate
def actual_function(x):
    return torch.sin(x)

### GRAPH ACTUAL FUNCTION

# precise data is used to create a reasonably precise simulation
# of the actual function using a fairly dense set of points
num_of_points = 50
precise_x = torch.linspace(MIN_X, MAX_X, num_of_points).unsqueeze(1)
actual_y = actual_function(precise_x)

def display_actual_function():
    plt.clf()
    plt.plot(precise_x,actual_y)
    plt.ylim(MIN_Y,MAX_Y)

### GENERATE DATA

train_x: torch.Tensor = None
train_y: torch.Tensor = None

valid_x: torch.Tensor = None
valid_y: torch.Tensor = None

class BasicDataset(Dataset):

    def __init__(self,x_tensor,y_tensor):
        super().__init__()
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor

    def __getitem__(self, index):
        return (self.x_tensor[index], self.y_tensor[index])

    def __len__(self):
        return len(self.x_tensor)

train_dataset: BasicDataset = None
valid_dataset: BasicDataset = None

def add_noise(y_tensor,noise_level):
    noise = torch.randn(y_tensor.size()) * noise_level
    return y_tensor + noise

MAX_POINTS = 200

def create_data():
    global train_x, train_y, valid_x, valid_y, train_dataset, valid_dataset

    data_slider_value = 9
    uniform_distribution = True  # currently only supporting random distribution
                                 # must leave True
    percent_points = 1 - (data_slider_value * 0.1)
    number_of_points = int(MAX_POINTS * percent_points)

    if uniform_distribution:
        data_x = torch.linspace(MIN_X, MAX_X, number_of_points).unsqueeze(1)

    data_y_original = actual_function(data_x)

    data_y = add_noise(data_y_original, 0.1)

    train_amount = int(number_of_points * 0.7)
    valid_amount = number_of_points - train_amount

    indices = torch.randperm(number_of_points)
    train_indices = indices[:train_amount]
    valid_indices = indices[train_amount:]

    train_x = data_x[train_indices]
    train_y = data_y[train_indices]

    valid_x = data_x[valid_indices]
    valid_y = data_y[valid_indices]

    train_dataset = BasicDataset(train_x, train_y)
    valid_dataset = BasicDataset(valid_x, valid_y)

def plot_data():
    plt.clf()
    plt.plot(precise_x,actual_y)
    plt.scatter(train_x,train_y, s=16, c='blue', marker='x', label='Training')
    plt.scatter(valid_x,valid_y, s=16, c='red', marker='d', label='Validation')
    plt.legend()
    plt.ylim(MIN_Y,MAX_Y)
    plt.show()

### GUI FOR DATA SELECTION

data_plot_output = Output()
choose_data_points_button = Button(description="Regenerate Data", button_style='info')

def display_data_selection():
    display(VBox([choose_data_points_button, data_plot_output]))
    regenerate_data(None)

def regenerate_data(_):  # will receive button as parameter when used as an event handler
    create_data()
    with data_plot_output:
        data_plot_output.clear_output(wait=True)
        plot_data()

choose_data_points_button.on_click(regenerate_data)

### NEURAL NETWORK DEFINITION

class ComplexNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        layer_list = [nn.Linear(1,40)]

        for _ in range(8):
            layer_list.append(nn.Linear(40,40))
            layer_list.append(nn.ReLU())

        layer_list.append(nn.Linear(40,1))

        self.layers = nn.Sequential(*layer_list)

    def forward(self,network_inputs):
        return self.layers(network_inputs)

### NETWORK TRAINING

network = ComplexNetwork()

import torch.optim as optim

# I'm actually pushing to overfit so the students can study this phenomenon
learning_rate = 0.01  # lower-learning rates are more prone to overfitting
batch_size = 2  # smaller-batch sizes are more prone to overfitting
loss_func = nn.MSELoss()
total_epochs = 20000
reporting_rate = 500
graphing_rate = 100

min_valid_state: Dict[str, torch.Tensor] = None
final_state: Dict[str, torch.Tensor] = None

def train_network(progress_bar, label_curr_progress, best_epoch_html):
    global min_valid_state, final_state, network

    network = ComplexNetwork()

    optimizer = optim.SGD(network.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    best_valid_epoch = -1
    min_valid_state = network.state_dict()
    min_valid_loss = float('inf')

    best_train_epoch = -1
    min_train_loss = float('inf')

    train_loss_history = []
    valid_loss_history = []

    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        train_sample_count = 0
        train_loss_sum = 0

        network.train()
        for sample_batch, label_batch in train_loader:
            predictions_batch = network(sample_batch)

            optimizer.zero_grad()
            loss = loss_func(predictions_batch, label_batch)
            loss.backward()

            optimizer.step()

            train_loss_sum += loss.item() * len(sample_batch)
            train_sample_count += len(sample_batch)

        epoch_train_loss = train_loss_sum / train_sample_count
        train_loss_history.append(epoch_train_loss)

        if epoch_train_loss < min_train_loss:
            best_train_epoch = epoch
            min_train_loss = epoch_train_loss

        valid_sample_count = 0
        valid_loss_sum = 0
        network.eval()
        with torch.no_grad():
            for sample_batch, label_batch in valid_loader:
                predictions_batch = network(sample_batch)

                loss = loss_func(predictions_batch, label_batch)

                valid_loss_sum += loss.item() * len(sample_batch)
                valid_sample_count += len(sample_batch)

        epoch_valid_loss = valid_loss_sum / valid_sample_count
        valid_loss_history.append(epoch_valid_loss)

        if epoch_valid_loss < min_valid_loss:
            best_valid_epoch = epoch
            min_valid_loss = epoch_valid_loss
            min_valid_state = copy.deepcopy(network.state_dict())

        if epoch % reporting_rate == 0:
            print(f"epoch {epoch:>5}: train loss: {epoch_train_loss:.5f}, valid loss: {epoch_valid_loss:.5f}")

        progress_percent = int(100 * epoch / total_epochs)

        best_epoch_html.value = f"<b>best training epoch: {best_train_epoch}</b>, " \
                                   f"loss: {min_train_loss:.5f}<br>" \
                                   f"<b>best validation epoch: {best_valid_epoch}</b>, " \
                                   f"loss: {min_valid_loss:.5f}<br>"

        progress_bar.value = progress_percent
        label_curr_progress.value = f"{progress_percent}%"

    end_time = time.time()

    print(f"Training Time: {(end_time - start_time):.1f} seconds")

    final_state = copy.deepcopy(network.state_dict())

### GUI FOR TRAINING

progress_bar = widgets.IntProgress(value=0, min=0, max=100, description='Progress:', bar_style='info', orientation='horizontal')
label_curr_progress = HTML("0%")
train_button = Button(description="Train Network", button_style='info')
loss_output = Output()
best_epoch_html = HTML()

def display_train_network():
    display(VBox([HBox([train_button,progress_bar,label_curr_progress]), loss_output, best_epoch_html]))

def run_training(_):
    best_epoch_html.value = ""
    with loss_output:
        clear_output(wait=True)
        train_network(progress_bar, label_curr_progress, best_epoch_html)

train_button.on_click(run_training)

### GRAPH RESULTS

def display_graph_network():
    num_of_points = 50
    graphing_x = torch.linspace(MIN_X, MAX_X, num_of_points).unsqueeze(1)

    network.load_state_dict(final_state)
    network.eval()
    with torch.no_grad():
        graphing_y = network(graphing_x)

    network.load_state_dict(min_valid_state)
    network.eval()
    with torch.no_grad():
        best_y = network(graphing_x)

    plt.clf()

    plt.plot(graphing_x, best_y, c='green', label="Best State")
    plt.plot(graphing_x, graphing_y, c='cyan', label="Final State")
    plt.plot(precise_x, actual_y, label="Original Function")
    plt.scatter(train_x, train_y, s=16, c='blue', marker='x', label='Training')
    plt.scatter(valid_x, valid_y, s=16, c='red', marker='d', label='Validation')
    plt.legend()
    plt.ylim(MIN_Y,MAX_Y)

### DISPLAY NETWORK

def display_curr_network_info():
    print(network)