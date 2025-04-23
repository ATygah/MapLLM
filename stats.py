################################################################################
#                                                                              #
#              MODEL STATISTICS & CONFIGURATION DETAILS                      #
#                                                                              #
# This file provides an overview of how various model parameters affect the     #
# overall model size, as well as detailing the contributions of different      #
# layers in the model architecture.                                            #
#                                                                              #
# Model Parameters:                                                            #
#   - num_layers                                                               #
#   - num_heads                                                                #
#   - embed_dim                                                                #
#   - vocab_size                                                               #
#   - batch_size                                                               #
#   - expansion_factor                                                         #
#   - context_length                                                           #
#   - dropout                                                                  #
#   - dtype                                                                    #
#   - activation_type                                                          #
#   - activation_function                                                      #
#                                                                              #
# Model Layers:                                                                #
#   - token_embedding                                                          #
#   - positional_embedding                                                     #
#   - transformer_block                                                        #
#   - feed_forward                                                             #
#   - output_head                                                              #
#   - conv_projection                                                          #
#   - shortcut                                                                 #
#                                                                              #
# Attention Types:                                                             #
#   - self_attention                                                           #
#   - cross_attention                                                          #
#                                                                              #
################################################################################

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap


######## PLOTTING THE DATA OF DIFFERENT LLM MODELS ########

# Load data from YAML
with open("info/llmContextLengths.yaml", "r") as f:
    data = yaml.safe_load(f)

df = pd.DataFrame(data["models"])

# Reorder columns
columns_titles = ["name", "context_length", "parameters", "model_size", "release_date"]
df = df.reindex(columns=columns_titles)

# Rename columns for better display
df.rename(
    columns={
        "name": "Model",
        "context_length": "Context Length",
        "parameters": "Parameters",
        "model_size": "Model Size",
        "release_date": "Release Date",
    },
    inplace=True,
)

# Compute numeric values for coloring without adding extra columns to the DataFrame
context_length_numeric = pd.to_numeric(df["Context Length"], errors="coerce")

def convert_parameters(param_str):
    try:
        param_str = param_str.strip(" ~")
        if '-' in param_str:
            value_str = param_str.split('-')[-1]
        else:
            value_str = param_str
        if value_str.endswith("M"):
            return float(value_str[:-1]) * 1e6
        elif value_str.endswith("B"):
            return float(value_str[:-1]) * 1e9
        elif value_str.endswith("T"):
            return float(value_str[:-1]) * 1e12
        else:
            return float(value_str)
    except Exception:
        return None

parameters_numeric = df["Parameters"].apply(convert_parameters)

# Column definitions with updated color schemes:
col_defs = [
    ColumnDefinition(
        name="Model", textprops={"ha": "left", "weight": "bold"}, width=2.5
    ),
    ColumnDefinition(
        name="Context Length",
        textprops={"ha": "center"},
        width=1.5,
        #cmap=normed_cmap(
        #    context_length_numeric.dropna(),
        #    cmap=matplotlib.cm.get_cmap("YlGnBu"),  # Lighter, smoother gradient
        #    num_stds=2, #adjusted std
        #),
    ),
    ColumnDefinition(
        name="Parameters",
        textprops={"ha": "center"},
        width=1.5,
        #cmap=normed_cmap(
        #    parameters_numeric.dropna(),
        #    cmap=matplotlib.cm.get_cmap("Reds"), # A nice red gradient, for emphasis
        #    num_stds=2.5, #adjusted std
        #),
    ),
    ColumnDefinition(name="Model Size", textprops={"ha": "center"}, width=1.5),
    ColumnDefinition(name="Release Date", textprops={"ha": "center"}, width=1.5),
]

# Set font and figure properties
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["savefig.bbox"] = "tight"

# Create the table
fig, ax = plt.subplots(figsize=(18, len(df) * 0.5 + 2))

table = Table(
    df,
    column_definitions=col_defs,
    row_dividers=True,
    footer_divider=True,
    ax=ax,
    textprops={"fontsize": 12},
    row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
    col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
    column_border_kw={"linewidth": 1, "linestyle": "-"},
    showindex=False  # Removed the index column from display
)

# Add header and subtitle
header_text = "\n LLM Context Lengths and Model Details (Smallest to Largest)"
header_props = {"fontsize": 16, "fontweight": "bold", "va": "center", "ha": "center"}
plt.text(0.5, 0.92, header_text, transform=fig.transFigure, **header_props)

subtitle_text = (
    "\n A comparison of Large Language Model specifications, ordered by approximate model size."
)
subtitle_props = {"fontsize": 12, "va": "center", "ha": "center", "color": "gray"}
plt.text(0.5, 0.90, subtitle_text, transform=fig.transFigure, **subtitle_props)

# Add footer
footer_text = "Source: Publicly available information"
footer_props = {"fontsize": 10, "va": "center", "ha": "center"}
plt.text(0.5, 0.08, footer_text, transform=fig.transFigure, **footer_props)

# Save the figure
fig.savefig("plots/llm_table.png", facecolor=ax.get_facecolor(), dpi=200)

print("Table saved as llm_table.png")


######## PLOTTING THE DATA ON THE IMPACT OF SEQUENCE LENGTH ON MODEL SIZE ########
