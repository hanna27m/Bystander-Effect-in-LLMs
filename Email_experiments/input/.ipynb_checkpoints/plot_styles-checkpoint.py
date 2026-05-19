import matplotlib.pyplot as plt

family_colors = {
    # Gemma (Orange/Gold) 
    "gemma12": "#E69F00",   # Rich Gold 
    "gemma27": "#D55E00",   # Burnt Orange

    # Qwen (Green/Teal) 
    "qwen8":   "#80CDC1",   # Muted Teal
    "qwen14":  "#018571",   # Deep Pine

    # Ministral (Purple/Magenta) 
    "ministral8":  "#CC79A7",  # Reddish Purple
    "ministral14": "#7A0177",  # Dark Royal Purple

    # LLaMA (Blue)
    "llama8":  "#0072B2"    # Strong Cobalt Blue
}

def set_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{mathptmx}"
    })

model_order = [
    "gemma12", "gemma27", "qwen8", "qwen14",  "ministral8", "ministral14", "llama8"
][::-1]

# Standard Thesis Dimensions of Plots
FULL_WIDTH = 6.0
HALF_WIDTH = 3.0
GOLDEN_RATIO = 0.618  