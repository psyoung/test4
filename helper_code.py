### CS106EA Exploring Artificial Intelligence
#   Patrick Young, Stanford University
#
#   This code is mean to be loaded by Colab Notebook
#   It's purpose is to hide most of the implementation code
#   making the Colab Notebook easier to follow.

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

### GUI STUFF

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