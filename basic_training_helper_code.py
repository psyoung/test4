### CS106EA Exploring Artificial Intelligence
#   Patrick Young, Stanford University
#
#   This code is mean to be loaded by Colab Notebook
#   It's purpose is to hide most of the implementation code
#   making the Colab Notebook easier to follow.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML

from IPython.display import Javascript

from dataclasses import dataclass

from typing import Tuple

try:
    import google.colab
    running_on_colab = True
except ImportError:
    running_on_colab = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

N_SAMPLES = 500
NOISE = 0.1

from sklearn.datasets import make_moons, make_classification, make_blobs
import matplotlib.pyplot as plt

features: np.ndarray = None  # sample features (x, y) floats -- location of sample in 2D space
labels: np.ndarray = None    # sample label (stored as an array on int with values 0 or 1)

dataset_choice = "Interleaved"

def make_data():
    global features, labels
    if dataset_choice == "Interleaved":
        features, labels = make_moons(n_samples=N_SAMPLES, noise=NOISE)
    else:
        range_min, range_max = -2, 2
        centers = np.random.uniform(low=range_min, high=range_max, size=(2, 2))
        features, labels = make_blobs(n_samples=N_SAMPLES, centers=centers, n_features=2,
                                      cluster_std=0.3)
        # features, labels = make_classification(n_samples=N_SAMPLES, n_features=2, n_classes=2,
        #                                        n_informative = 2,
        #                                        n_redundant = 0,
        #                                        class_sep = 2,
        #                                        n_clusters_per_class=1)

def plot_data():
    plt.clf()
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

class BasicDataset(Dataset):
    def __init__(self, features_data, labels_data):
        super().__init__()
        self.features_data = torch.from_numpy(features_data).float()

        # unsqueeze y since PyTorch will expect a vector of outputs
        self.labels_data = torch.from_numpy(labels_data).float().unsqueeze(1)

    def __getitem__(self, index):
        return (self.features_data[index], self.labels_data[index])

    def __len__(self):
        return len(self.features_data)

train_dataset: BasicDataset = None
# valid_dataset: BasicDataset = None
# test_dataset: BasicDataset = None

def make_datasets():
    global train_dataset, valid_dataset, test_dataset

    data_count = len(features)
    train_count = int(data_count * 1)
    # valid_count = int(data_count * 0.15)
    # test_count = data_count - train_count - valid_count

    indices = np.arange(data_count)
    np.random.shuffle(indices)
    shuffled_features = features[indices].copy()
    shuffled_labels = labels[indices].copy()

    train_features = shuffled_features[:train_count]
    train_labels = shuffled_labels[:train_count]

    # valid_features = shuffled_features[train_count:train_count+valid_count]
    # valid_labels = shuffled_labels[train_count:train_count+valid_count]

    # test_features = shuffled_features[train_count+valid_count:]
    # test_labels = shuffled_labels[train_count+valid_count:]

    train_dataset = BasicDataset(train_features,train_labels)
    # valid_dataset = BasicDataset(valid_features,valid_labels)
    # test_dataset = BasicDataset(test_features,test_labels)

def make_new_data():
    make_data()
    # plot_data()
    make_datasets()

make_new_data()

import torch
import torch.nn as nn

class MyLinearRegressionNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2,1),
        )

    def forward(self,input_batch: torch.Tensor)->torch.Tensor:
        return torch.sigmoid(self.layers(input_batch))

class MyBasicNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(2,10), nn.ReLU(),
                nn.Linear(10,10), nn.ReLU(),
                nn.Linear(10,1)
        )

    def forward(self,input_batch: torch.Tensor)->torch.Tensor:
        return torch.sigmoid(self.layers(input_batch))

def draw_graph(title=None,draw_model_predictions:bool = False):
    # Plot the data
    if title is not None:
        fig = plt.gcf()
        fig.suptitle(title)

    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', edgecolor='k', s=20)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    if draw_model_predictions:
        # Create a mesh grid
        h = .02  # Step size in the mesh
        x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
        y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Predict the function value for the whole grid
        grid_data = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
        with torch.no_grad():
            Z = network(grid_data)
        Z = Z.cpu().numpy().reshape(xx.shape)

        # Plot the decision boundary by setting alpha to 0.8
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, cmap='viridis')
        plt.colorbar()

        # Optionally, plot the contour line for the decision boundary at 0.5
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    plt.show()

### GUI

# my_button_style = {
#     'button_color': 'lightblue',
#     'font_weight': 'bold',
#     'font_size': '12pt'
# }

data_options = ["Blobs","Interleaved"]

data_label = HTML("<b>Data Choices:</b>", layout=widgets.Layout(width="90px"))
data_dropdown = widgets.Dropdown(options=data_options,value=dataset_choice, layout=widgets.Layout(width="160px"))
data_generate_button = widgets.Button(description="Regenerate Data", button_style='info')


data_output = widgets.Output()

def draw_graph_to_output(output: widgets.Output, draw_model_predictions=False, title=None):
    with output:
        output.clear_output(wait=True)
        draw_graph(draw_model_predictions=draw_model_predictions,title=title)

draw_graph_to_output(data_output, False)

def change_data(_):
    global dataset_choice
    dataset_choice = data_dropdown.value
    make_data()
    make_datasets()
    draw_graph_to_output(data_output, draw_model_predictions=False)

data_dropdown.observe(change_data, names='value')
data_generate_button.on_click(change_data)

def display_choose_data():
    display(VBox([
                HBox([data_label, data_dropdown, data_generate_button]),
                data_output]))

import torch.optim as optim

##  @dataclass
# class NetworkandInfo:
#     network: nn.Module
#     optimizer: optim

LEARNING_RATE = 0.05
BATCH_SIZE = 10

linear_regression = True
network: nn.Module = None
optimizer: optim = None
loss_fn = nn.BCELoss()

epoch: int = 0

train_loader: DataLoader = None

def reset_network():
    change_network()
    # print("reset_network")
    # global optimizer, epoch, train_loader
    # epoch = 0
    # network.reset_parameters()
    # optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def change_network(network_type="Linear Regression"):
    # print("change_network:" + network_type)
    global network, optimizer, epoch, train_loader
    epoch = 0

    if network_type == "Linear Regression":
        network = MyLinearRegressionNetwork()
    else:
        network = MyBasicNetwork()
    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def run_epochs(n:int = 100, graphing_rate:int=1, output: widgets.Output=None):
    global epoch
    for count in range(n):
        network.train()

        for (features_batch, label_batch) in train_loader:
            (features_batch, label_batch) = (features_batch.to(device), label_batch.to(device))
            predictions_batch = network(features_batch)

            loss = loss_fn(predictions_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch += 1

        # always print first epoch (but not first count) and the last epoch in the training run
        if epoch == 1 or epoch % graphing_rate == 0 or count == n - 1:
            if output is None:
                draw_graph(draw_model_predictions=True,title=f"Epoch: {epoch}")
            else:
                with output:
                    draw_graph(draw_model_predictions=True,title=f"Epoch: {epoch}")

change_network("Basic Neural Network")

model_label = HTML("<b>Choose Model:</b>", layout=widgets.Layout(width="90px"))  # Dropdown description gets cut off
model_options = ["Linear Regression", "Basic Neural Network"]
model_dropdown = widgets.Dropdown(options=model_options, value="Basic Neural Network",
                                  layout=widgets.Layout(width="170px"))

model_reset_button = widgets.Button(description="Model Reset", button_style='info')

epochs_label = HTML("<b>How many Epochs to Run:</b>", layout=widgets.Layout(width="160px"))
epochs_text = widgets.Text(value="200", layout=widgets.Layout(width="60px"))

graph_freq_start = HTML("<b>Graph Frequency: </b>", layout=widgets.Layout(width="110px"))
graph_freq_text = widgets.Text(value="10", layout=widgets.Layout(width="60px"))
graph_freq_end = HTML("&nbsp;&nbsp;&nbsp;How many Epochs to run per graph generated")

train_button = widgets.Button(description="Start Training", button_style='info')
def set_train_button_label(label: str):
    train_button.description = label

# results_output = widgets.Output(layout=widgets.Layout(height="400px", overflow='auto'))
results_output = widgets.Output(layout=widgets.Layout())

def change_model(_):
    change_network(model_dropdown.value)
    with results_output:
        results_output.clear_output()
    set_train_button_label("Start Training")

model_dropdown.observe(change_model, names='value')
model_reset_button.on_click(change_model)

def display_choose_model_and_train():
    display(VBox([HBox([model_label,model_dropdown,model_reset_button]),
              HBox([epochs_label,epochs_text]),
              HBox([graph_freq_start,graph_freq_text,graph_freq_end]),
              train_button,
              results_output]))

def train(_):
    try:
        results_output.clear_output(wait=True)
        set_train_button_label("Continue Training")
        epochs_to_run = int(epochs_text.value)
        reporting_rate = int(graph_freq_text.value)
        # print(f"{epochs_to_run}: {reporting_rate}")
        run_epochs(epochs_to_run,reporting_rate,results_output)
    except ValueError:
        print(f"Not a Number")

train_button.on_click(train)

def display_curr_network_info():
    print(network)