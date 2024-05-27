### CS106EA Exploring Artificial Intelligence
#   Patrick Young, Stanford University
#
#   This code is mean to be loaded by Colab Notebook
#   It's purpose is to hide most of the implementation code
#   making the Colab Notebook easier to follow.

### IMPORTS AND BASIC SETUP

import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, HTML, Label
from IPython.display import clear_output

import sys
import io
from contextlib import redirect_stdout

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim

import matplotlib.pyplot as plt

import numpy as np
import time

from collections import OrderedDict

from dataclasses import dataclass, field
from typing import Type, Callable, Tuple, Dict
import math
import random

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

### SETUP CSS STYLES
#   These will only be used if you if you explicitly include
#   them in a given "display()" call

html_style = HTML(
    value="""
<style>
.control-major-label {
    font-size: 1.2em;
    font-weight: bold;
}
.control-label {
    font-size: 1em;
    font-weight: bold;
}
.control-minor-label {
    font-size: 0.9em;
}
.widget-checkbox {
    width: auto !important;  /* Adjust this if necessary */
    /*border: 1px solid blue;*/ /* To see the actual space taken by the checkbox container */
}
.widget-checkbox > label {
    margin: 0 !important;
    padding: 0 !important;
    width: auto !important;
    /*border: 1px solid red;*/ /* To see the space taken by the label */
}
.widget-checkbox input[type="checkbox"] {
    margin: 0 !important;
}
.widget-inline-hbox .widget-label {
    flex: 0 0 auto !important;
}
.widget-inline-hbox {
    align-items: center; /* Align items vertically in the center */
    min-width: 0; /* Helps prevent flex containers from growing too large */
}
.code {
    font-family: 'Courier New', Courier, monospace;
    font-weight: bold;
    line-height: 0.5;
    margin: 0;
    padding: 0;
}
</style>

    """
)

### DEFINE FUNCTIONS
#   These are the functions that we will try to approximate

def flat(x:np.ndarray):
    return np.ones_like(x)

def slope(x:np.ndarray):
    return np.copy(x)

def relu(x: np.ndarray):
    return np.maximum(0,x)

def square(x:np.ndarray):
    return np.square(x)

def sine(x:np.ndarray):
    return np.sin(x)

def non_np_square(x: float)->float: # unused, just wrote this to make sure I understood the implementation
    x_modulo = x % 2
    if x_modulo < 1:
        return 0.5
    else: return -0.5

def square_wave(x:np.ndarray)->np.ndarray:
    x_modulo = x % 2
    return np.where(x_modulo < 1, 0.5, -0.5)

def non_np_step(x:float)->float:
    x_floor = math.floor(x)
    return x_floor + 0.5

def stairstep(x:np.ndarray)->np.ndarray:
    x_floor = np.floor(x)
    return x_floor + 0.5

@dataclass
class FunctionInfo:
    name: str
    func: Callable[[np.ndarray], np.ndarray]

function_list = [FunctionInfo("Flat",flat),
                 FunctionInfo("Slope",slope),
                 FunctionInfo("ReLU",relu),
                 # FunctionInfo("Squared",square),
                 FunctionInfo("Sine Wave",sine),
                 FunctionInfo("Square Wave",square_wave),
                 FunctionInfo("Stairstep",stairstep)]

### GUI CODE FOR SELECTING FUNCTION

function_options = {func.name: func for func in function_list}
function_label = HTML(value="<span class='control-major-label'>Function:</span>")

function_dropdown = widgets.Dropdown(
        options = function_options,
        layout=Layout(width="100px"),
        value=function_options["Sine Wave"]
)

drop_down_box = HBox([function_label, function_dropdown])

STEP=0.2
slider_x_range = widgets.FloatSlider(value=6*math.pi,min=2.0, max=8*math.pi,step=STEP,
                                    readout=False,
                                    layout=Layout(width="200px"))

label_for_range_slider = HTML(value="<span class='control-label'>Input Range:</span>",
                              layout=Layout(min_width="80px"))
output_for_range_slider = Label(value="")

slider_x_range_section = HBox([label_for_range_slider,slider_x_range,output_for_range_slider],
                              layout=Layout(margin="0px 10px"))

btn_draw = widgets.Button(description="Draw Graph")

#controls_section = HBox([drop_down_box, slider_x_range_section, btn_draw])
controls_section = HBox([drop_down_box, slider_x_range_section])
output = widgets.Output()
original_func_box = VBox([controls_section,output])

GRAPHING_STEP = 0.01  # determines how many points to use when graphing function
                      # unless it's set pretty high stairset function looks pretty bad
                      # 0.1 is fine without the stairstep

def draw_graph(_):
    with output:
        clear_output(wait=True)
        canvas = plt.figure()
        curr_func_info: FunctionInfo = function_dropdown.value

        x_numpy = np.arange(-slider_x_range.value / 2.0, slider_x_range.value / 2.0 , step=GRAPHING_STEP)  # Example range and step
        y_numpy = curr_func_info.func(x_numpy)  # Example function
        plt.plot(x_numpy, y_numpy)
        # plt.set_title('Sine Wave')
        plt.show()

def slider_updated(_): # ignore widget passed in
    draw_graph(None)

slider_x_range.observe(slider_updated,names=['value'])
function_dropdown.observe(draw_graph,names=['value'])

btn_draw.on_click(draw_graph)

# with output:
#     clear_output(wait=True)
#     canvas = plt.gcf()
#     display(canvas)

draw_graph(None)

def display_choose_graph_ui():
    display(html_style,original_func_box)

### GUI FOR CHOOSING DATA POINTS

# Define plot size in inches (width, height) and DPI
plot_width, plot_height, dpi = 5.5, 5.0, 100

## Slider to choose how much data will be used
slider_amount_of_data = widgets.IntSlider(
    value=3,
    min=1,
    max=5,
    step=1,
    readout=False,
    layout=Layout(width="150px")
)
label_amount_of_data_main = HTML(f"<span class='control-major-label'>Amount of Data:</span>")
label_min_amount_of_data = HTML(f"<span class='control-minor-label'>Very Little</span>")
label_max_amount_of_data = HTML(f"<span class='control-minor-label'>A Lot</span>")

box_amount_of_data = HBox([label_amount_of_data_main,
                           label_min_amount_of_data, slider_amount_of_data, label_max_amount_of_data])

## Radio Buttons to choose whether the samples should be uniformly or randomly
#  distributed along the X-axis
distribution_choices = ['Random','Uniform']
radio_uniform_random = widgets.RadioButtons(
    options = distribution_choices,
    value = distribution_choices[0],
    disabled = False,
    layout=Layout(margin='6px 0px 0px 0px')
)
label_distribution_choices = widgets.HTML(
    f"<span class='control-major-label' style='margin: 0px 10px 0px 0px;'>Distribution Type:</span>")

box_distribution_choices = HBox([label_distribution_choices, radio_uniform_random],
                                layout=Layout(align_items='flex-start'))

## Stack Data Amount with Distribution Type and Add Choose Data Button

choices_boxes = HBox([box_amount_of_data, box_distribution_choices],layout=Layout(justify_content='space-between'))

btn_pick_data = widgets.Button(description="Choose Data",layout=Layout(margin="0px 20px 0px 0px"))
btn_draw_data = widgets.Button(description="Draw Data",layout=Layout(margin="0px 20px 0px 0px"))
btn_clear_graph = widgets.Button(description="Clear Graph",layout=Layout(margin="0px"))

# data_choice_buttons = HBox([btn_pick_data, btn_draw_data, btn_clear_graph])
data_choice_buttons = HBox([btn_pick_data])

full_data_controls = VBox([choices_boxes, data_choice_buttons], layout=Layout(width="800px"))

## Add Acutal Output and Selection for Data to Plot

output_data_selection = widgets.Output(
    layout=widgets.Layout(width=f"{plot_width * dpi}px", height=f"{plot_height * dpi}px")
)
# output_data_selection = widgets.Output(layout=Layout(width="550px"))

with output_data_selection:
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)
    ax.text(0.5, 0.5, 'Your plot will appear here\n"Choose Data" options\nthen click on "Draw Data"', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    plt.axis('off')  # Turn off the axis
    plt.show()

border_blue_test = Layout(border='1px solid blue')
border_red_test = Layout(border='1px solid red')
check_chart_original = widgets.Checkbox(value=True)
check_chart_train = widgets.Checkbox(value=True)
check_chart_valid = widgets.Checkbox(value=True)
check_chart_test = widgets.Checkbox(value=True)

label_chart_overall = HTML("<span class='control-major-label'>Items to Display</span>")
label_chart_original = HTML("<span class='control-label'>Actual Function</span>")
label_chart_train = HTML("<span class='control-label'>Training Data</span>")
label_chart_valid = HTML("<span class='control-label'>Validation Data</span>")
label_chart_test = HTML("<span class='control-label'>Final Testing Data</span>")

chart_controls = VBox([
    label_chart_overall,
    HBox([check_chart_original, label_chart_original],layout=Layout(justify_content='flex-start')),
    HBox([check_chart_train, label_chart_train],layout=Layout(justify_content='flex-start')),
    HBox([check_chart_valid, label_chart_valid],layout=Layout(justify_content='flex-start')),
    HBox([check_chart_test, label_chart_test],layout=Layout(justify_content='flex-start'))])

data_selection_cell_box = VBox([full_data_controls,
                                HBox([output_data_selection, chart_controls])])

### ACTUAL COMPUTING

## Define Dataset
class BasicDataset (Dataset):
    def __init__(self, x_np: np.ndarray, y_np: np.ndarray, name: str=""):
        super().__init__()
        self.x_torch = torch.from_numpy(x_np[:, np.newaxis]).float().to(device)
        self.y_torch = torch.from_numpy(y_np)[:, None].float().to(device)
        self.name = name

    def __getitem__(self,index)->Tuple[torch.Tensor, torch.Tensor]:
        return (self.x_torch[index], self.y_torch[index])

    def get_numpy_arrays(self)-> Tuple[np.ndarray, np.ndarray]:
        return (self.x_torch.cpu().numpy(), self.y_torch.cpu().numpy())

    def __len__(self):
        return len(self.x_torch)

## Actual Selection and Graphing

FULL_DATA_COUNT = 100

train_dataset: BasicDataset = None
valid_dataset: BasicDataset = None
test_dataset: BasicDataset = None

def choose_data(_):  # event handler for button, but we don't actually need the button parameter
    global train_dataset, valid_dataset, test_dataset

    # Determine how much data to use and the range of the data
    percent_to_use = 100 - (5 - slider_amount_of_data.value) * 20
    num_data_points = int(percent_to_use)

    max_x = slider_x_range.value / 2.0
    min_x = -slider_x_range.value / 2.0

    # Check on user choice for Random or Uniform Distribution
    if radio_uniform_random.value == 'Random':
        temp_data_x = np.random.uniform(min_x,max_x,num_data_points)
    else:
        temp_data_x = np.linspace(min_x,max_x,num_data_points)

    # Calculate actual data based on previous Function Dropdown
    curr_func_info: FunctionInfo = function_dropdown.value
    temp_data_y = curr_func_info.func(temp_data_x)

    # Randomize Data
    indices = np.arange(num_data_points)
    np.random.shuffle(indices)

    full_data_x = temp_data_x[indices].copy()
    full_data_y = temp_data_y[indices].copy()

    # Break data into Train, Valid, and Test Datasets
    total_samples = len(full_data_x)
    num_train = int(0.7 * total_samples)
    num_valid = int(0.15 * total_samples)
    num_test = total_samples - num_train - num_valid

    train_dataset = BasicDataset(
                        full_data_x[:num_train], full_data_y[:num_train], "Training")
    valid_dataset = BasicDataset(
                        full_data_x[num_train:num_train+num_valid], full_data_y[num_train:num_train+num_valid], "Validation")
    test_dataset = BasicDataset(
                        full_data_x[num_train+num_valid:], full_data_y[num_train+num_valid:], "Testing")

    graph_data(None)

def graph_data(_): # ignore widget passed in when called as an event handler
    with output_data_selection:
        clear_output(wait=True)
        display(HTML(""))
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)

        if check_chart_train.value:
            (x_np, y_np) = train_dataset.get_numpy_arrays()
            ax.scatter(x_np, y_np, s=16, c='blue', marker='x', label='Train')
        if check_chart_valid.value:
            (x_np, y_np) = valid_dataset.get_numpy_arrays()
            ax.scatter(x_np, y_np, s=16, c='green', marker='o', label='Validation')
        if check_chart_test.value:
            (x_np, y_np) = test_dataset.get_numpy_arrays()
            ax.scatter(x_np, y_np, s=16, c='red', marker='d', label='Final Testing')

        if check_chart_original.value:
            curr_func_info: FunctionInfo = function_dropdown.value
            x_numpy = np.arange(-slider_x_range.value / 2.0, slider_x_range.value / 2.0 , step=GRAPHING_STEP)  # Example range and step
            y_numpy = curr_func_info.func(x_numpy)  # Example function
            plt.plot(x_numpy, y_numpy,c='black',linewidth=1)

        if check_chart_train.value or check_chart_valid.value or check_chart_test.value:
            ax.legend()

        plt.show()

def clear_graph(_):
    with output_data_selection:
        # print(time.time())
        # print()
        clear_output(wait=True)
        display(HTML("<p>Output cleared.</p>"))

### BACK TO GUI

## Wire Everything Up and Display

btn_pick_data.on_click(choose_data)
btn_draw_data.on_click(graph_data)
btn_clear_graph.on_click(clear_graph)

check_chart_original.observe(graph_data, names='value')
check_chart_train.observe(graph_data, names='value')
check_chart_valid.observe(graph_data, names='value')
check_chart_test.observe(graph_data, names='value')

def display_choose_data_points():
    display(html_style,data_selection_cell_box)

### DEFINE ACTUAL NEURAL NETWORKS

class TinyNetwork(nn.Module):
    """
    A tiny network directly mapping inputs through an activation function to the output.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,1),
            nn.Tanh(),
            nn.Linear(1,1)
        )

    def forward(self, x_batch: torch.Tensor)->torch.Tensor:
        return self.layers(x_batch)

class SmallSingleLayerNetwork(nn.Module):
    """
    A simple network with a single hidden layer of five neurons.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,5),
            nn.Tanh(),
            nn.Linear(5,1)
        )

    def forward(self, x_batch: torch.Tensor)->torch.Tensor:
        return self.layers(x_batch)

class WideSingleLayerNetwork(nn.Module):
    """
    A network with a wide single hidden layer of 75 neurons.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,75),
            nn.Tanh(),
            nn.Linear(75,1))

    def forward(self,network_inputs):
        return self.layers(network_inputs)

class DeepNarrowNetwork(nn.Module):
    """
    A network with 20 hidden layers of only 5 neurons each.
    """

    def __init__(self):
        super().__init__()
        layer_list = [nn.Linear(1,5)]

        for i in range(20):
            layer_list.append(nn.Linear(5,5))
            layer_list.append(nn.ReLU())

        layer_list.append(nn.Linear(5,1))

        self.layers = nn.Sequential(*layer_list)

    def forward(self,network_inputs):
        return self.layers(network_inputs)

class BalancedNetwork(nn.Module):
    """
    A balanced network with 5 hidden layers of 20 neurons each.
    """

    def __init__(self):
        super().__init__()
        layer_list = [nn.Linear(1,20)]

        for _ in range(5):
            layer_list.append(nn.Linear(20,20))
            layer_list.append(nn.ReLU())

        layer_list.append(nn.Linear(20,1))

        self.layers = nn.Sequential(*layer_list)

    def forward(self,network_inputs):
        return self.layers(network_inputs)

@dataclass
class NetworkInfo:
    name: str
    network_class: Type[nn.Module]
    description: str  = field(default="", init=False)
    printout: str  = field(default="", init=False)  # actual printout of network
    # trained_parameters: Dict[str, torch.Tensor] = field(default_factory=dict) # cache parameters after training (but not for specific ranges)

    def __post_init__(self):
        self.description = self.network_class.__doc__
        temp_out = io.StringIO()
        with redirect_stdout(temp_out):
            print(self.network_class())
        self.printout = temp_out.getvalue()

networks_supported = [NetworkInfo("Tiny Network",TinyNetwork),
                      NetworkInfo("Small Hidden Layer Network",SmallSingleLayerNetwork),
                      NetworkInfo("Wide Hidden Layer Network",WideSingleLayerNetwork),
                      NetworkInfo("Deep and Narrow Network",DeepNarrowNetwork),
                      NetworkInfo("Balanced Network",BalancedNetwork),
                      ]

### DEFINE TRAINING ROUTINE

@dataclass
class TrainingParameters:
    max_epochs: int = 6000
    batch_size: int = 15
    optimizer_type: Type[torch.optim.Optimizer] =torch.optim.SGD
    loss_fn_type: Type[nn.Module] = nn.MSELoss
    learning_rate: float = 0.02
    use_patience: bool = True
    patience: int = 500
    shuffle_loaders: bool = False

@dataclass
class ReportingParameters:
    print_progress: bool = False
    progress_rate: int = 500  # how often in epochs to print a status update
    draw_graphs: bool = False
    graph_frequency: int = 500 # how often to print a graph
    progress_bar: widgets.IntProgress = None
    progress_text: widgets.HTML = None

@dataclass
class TrainingHistory:
    # use these to determine when to stop early and to recover best network state
    # after we've passed by until we hit patience
    min_valid_loss: float = field(default=float('inf'), init=False)
    min_valid_epoch: int = field(default=0, init=False)
    min_valid_state: OrderedDict[str, torch.Tensor] = field(default=None, init=False)
    valid_loss_history: list = field(default_factory=list, init=False)

    early_stopping: bool = field(default=False, init=False)

    # currently just tracking these for information on how our training is behaving
    # conjecture for larger networks we should see continual drop, for very small
    # networks these may go all over the place
    min_train_loss: float = field(default=float('inf'), init=False)
    min_train_epoch: int = field(default=0, init=False)
    train_loss_history: list = field(default_factory=list, init=False)

    training_time: float = field(default=0, init=False)

def train_network(network: nn.Module,
                  train_params: TrainingParameters,
                  report_params: ReportingParameters)->TrainingHistory:

    train_loader = DataLoader(train_dataset, batch_size=train_params.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=train_params.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=train_params.batch_size)
    history_data = TrainingHistory()

    epochs_since_improvement = 0  # used in conjunction with patience TrainingParameter
    early_stopping = False

    actual_optimizer = train_params.optimizer_type(
                network.parameters(), lr=train_params.learning_rate)
    actual_loss_fn = train_params.loss_fn_type()

    history_data.min_valid_state = network.state_dict()
    history_data.min_valid_loss = float('inf')

    if report_params.progress_bar is not None:
        progress_bar_update_rate = int(train_params.max_epochs / 100)

    start_time = time.time()

    for epoch in range(1,train_params.max_epochs+1):

        train_sample_count = 0
        train_loss_sum = 0
        network.train()
        for sample_batch, ground_truth_batch in train_loader:

            predictions_batch = network(sample_batch)

            actual_optimizer.zero_grad()
            loss = actual_loss_fn(predictions_batch, ground_truth_batch)
            loss.backward()

            actual_optimizer.step()

            train_loss_sum += loss.item() * len(sample_batch)
            train_sample_count += len(sample_batch)

        epoch_train_loss = train_loss_sum / train_sample_count
        history_data.train_loss_history.append(epoch_train_loss)

        if epoch_train_loss < history_data.min_train_loss:
            history_data.min_train_loss = epoch_train_loss
            history_data.min_train_epoch = epoch

        valid_sample_count = 0
        valid_loss_sum = 0
        network.eval()
        with torch.no_grad():
            for sample_batch, ground_truth_batch in valid_loader:

                predictions_batch = network(sample_batch)

                loss = actual_loss_fn(predictions_batch, ground_truth_batch)

                valid_loss_sum += loss.item() * len(sample_batch)
                valid_sample_count += len(sample_batch)

            epoch_valid_loss = valid_loss_sum / valid_sample_count
            history_data.valid_loss_history.append(epoch_valid_loss)

            if epoch_valid_loss < history_data.min_valid_loss:
                history_data.min_valid_loss = epoch_valid_loss
                history_data.min_valid_epoch = epoch
                history_data.min_valid_state = network.state_dict()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if train_params.use_patience and epochs_since_improvement > train_params.patience:
                    history_data.early_stopping = True
                    if report_params.progress_bar is not None:
                        progress_percent = int(100 * epoch / train_params.max_epochs)
                        report_params.progress_bar.value = progress_percent
                        report_params.progress_text.value = f"{progress_percent}% - <b><u>Stopped Early</u></b>"
                    break

        if report_params.progress_bar is not None and epoch%progress_bar_update_rate == 0:
            progress_percent = int(100 * epoch / train_params.max_epochs)
            report_params.progress_bar.value = progress_percent
            report_params.progress_text.value = f"{progress_percent}%"

        if report_params.print_progress and epoch%report_params.progress_rate == 0:
            print(f"epoch {epoch:>4}: train loss{epoch_train_loss}, valid loss{epoch_valid_loss}")
        if report_params.draw_graphs and epoch%report_params.graph_frequency == 0:
            graph_predicted_output_vs_actual_function(network, title=f"Epoch: {epoch}")

    network.load_state_dict(history_data.min_valid_state)

    end_time = time.time()
    history_data.training_time = end_time - start_time

    return history_data

### GRAPHING AND REPORTING FUNCTIONS

# older version that printed out to the entire MatPlotLib plt
def graph_predicted_output_vs_actual_function(network: nn.Module,
                                              *, title:str=None):

    x_numpy_actual = np.arange(-slider_x_range.value / 2.0, slider_x_range.value / 2.0 , step=GRAPHING_STEP)  # Example range and step
    torch_x_actual = torch.from_numpy(x_numpy_actual).float().to(device)
    torch_x_unsqueezed = torch_x_actual.unsqueeze(1)  # PyTorch expects samples to provide a vector of features, not just a single feature
    network.eval()
    with torch.no_grad():
        y_torch_actual = network(torch_x_unsqueezed).detach().to('cpu')
        y_numpy_actual = y_torch_actual.numpy()

    plt.plot(x_numpy_actual, y_numpy_actual,c='black',linewidth=1)

    (x_np, y_np) = train_dataset.get_numpy_arrays()
    plt.scatter(x_np, y_np, s=16, c='blue', marker='x', label='Train')

    (x_np, y_np) = valid_dataset.get_numpy_arrays()
    plt.scatter(x_np, y_np, s=16, c='green', marker='o', label='Validation')

    if title is not None:
        plt.title(title)

    plt.legend()

    plt.show()

# Newer version that just plots to a single set of axes
def graph_predicted_output_vs_actual_function_on_ax(network: nn.Module,
                                              ax,
                                              *, title:str=None):

    ax.clear()

    x_numpy_actual = np.arange(-slider_x_range.value / 2.0, slider_x_range.value / 2.0 , step=GRAPHING_STEP)  # Example range and step
    torch_x_actual = torch.from_numpy(x_numpy_actual).float().to(device)
    torch_x_unsqueezed = torch_x_actual.unsqueeze(1)  # PyTorch expects samples to provide a vector of features, not just a single feature
    network.eval()
    with torch.no_grad():
        y_torch_actual = network(torch_x_unsqueezed).detach().to('cpu')
        y_numpy_actual = y_torch_actual.numpy()

    ax.plot(x_numpy_actual, y_numpy_actual,c='black',linewidth=1)

    (x_np, y_np) = train_dataset.get_numpy_arrays()
    ax.scatter(x_np, y_np, s=16, c='blue', marker='x', label='Train')

    (x_np, y_np) = valid_dataset.get_numpy_arrays()
    ax.scatter(x_np, y_np, s=16, c='green', marker='o', label='Validation')

    if title is not None:
        ax.set_title(title)

    ax.legend()

def formatted_training_run_info(history_data: TrainingHistory)->widgets.HTML:
    html_string = f"<b>Training Time:</b> {history_data.training_time:.2f} Seconds<br>"
    html_string += f"<b>Speed:</b> {(len(history_data.train_loss_history)/history_data.training_time):.2f} Epochs/Seconds<br>"
    html_string += f"<b>Epochs Run:</b> {len(history_data.train_loss_history)} Epochs<br>"
    if history_data.early_stopping:
        html_string += f"<b>Stopped Early</b> -- Best Epoch was {history_data.min_valid_epoch}<br>"
    html_string += f"<b>Best Loss</b> {history_data.min_valid_loss:.6f}"
    return html_string

### GUI TO CHOOSE NEURAL NETWORK TO USE

network_options = {network.name: network for network in networks_supported}
network_label = HTML(value="<span class='control-major-label'>Network to Use:</span>")

network_dropdown = widgets.Dropdown(
        options = network_options,
        layout=Layout(width="200px"),
        value=network_options["Balanced Network"]
)

#network_description = HTML(value=f"<span style='margin-left: 15px'>{network_options['Balanced Network'].description}<span>")
network_description = HTML()
network_display_desc = HTML()

def on_network_change(change):
    network_description.value = f"<span style='margin-left: 15px'>{network_dropdown.value.description}<span>"
    display_html = (network_dropdown.value.printout).replace("\n","<br>")
    display_html = display_html.replace(" ","&nbsp;")
    network_display_desc.value = "<span class='code'>" + display_html + "</span>"
network_dropdown.observe(on_network_change, names='value')

on_network_change(None)  # force to pull values from original setting of pulldown

network_drop_down_box = HBox([network_label, network_dropdown,network_description])
network_choice_full_gui = VBox([network_drop_down_box,network_display_desc])

def display_choose_network_architecture():
    display(html_style,network_choice_full_gui)

### TRAIN NETWORK AND GRAPH

def create_and_train(network_info: NetworkInfo,
                     train_params: TrainingParameters,
                     report_params: ReportingParameters) -> Tuple[nn.Module, TrainingHistory]:
    network = network_info.network_class().to(device)

    return network, train_network(network,train_params,report_params)

def graph_results():
    with output_data_selection:
        clear_output(wait=True)
        display(HTML(""))

        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)

btn_train_network = widgets.Button(description='Train Network')

label_patience = HTML("<span class='control-label' style='margin-left: 20px'>Allow Early Stopping</span>")
check_patience = widgets.Checkbox(value=True)

label_progress = HTML("<span class='control-label' style='margin-right: 10px'>Progress</span>")
progress_bar = widgets.IntProgress(value=0, min=0, max=100, description='Progress:', bar_style='info', orientation='horizontal')
label_curr_progress = HTML("0%")

progress_section = HBox([progress_bar,label_curr_progress])

run_info_widget = HTML()
output_chart = widgets.Output(
    layout=widgets.Layout(width=f"{plot_width * dpi}px", height=f"{plot_height * dpi}px")
)
with output_chart:
    # fig, ax = plt.subplots()
    train_fig, train_ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)
    train_ax.text(0.5, 0.5, 'Your plot will appear here', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    plt.axis('off')  # Turn off the axis
    plt.show()

train_controls = HBox([btn_train_network, label_patience, check_patience])

train_contents = VBox([train_controls, progress_section, run_info_widget, output_chart])

def training_run(_):  # will be passed Button, ignore it
    run_info_widget.value="<b>Running ...</b>"
    with output_chart:
        clear_output(wait=True)  # Clear the output area
        train_fig, train_ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)
        train_ax.text(0.5, 0.5, 'Your plot will appear here', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        plt.axis('off')  # Turn off the axis
        plt.show()

    training_choices = TrainingParameters
    reporting_choices = ReportingParameters

    training_choices.use_patience = check_patience.value

    reporting_choices.progress_bar = progress_bar
    reporting_choices.progress_text = label_curr_progress

    (curr_network, curr_history) = create_and_train(network_dropdown.value,training_choices, reporting_choices)

    run_info_widget.value = formatted_training_run_info(curr_history)
    with output_chart:
        clear_output(wait=True)  # Clear the output area
        fig, train_ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)
        graph_predicted_output_vs_actual_function_on_ax(curr_network, train_ax)
        train_ax.legend()  # Ensure legend is displayed
        plt.show()

btn_train_network.on_click(training_run)

def display_train_network():
    display(html_style, train_contents)