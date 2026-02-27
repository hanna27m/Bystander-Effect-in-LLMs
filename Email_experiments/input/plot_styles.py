import matplotlib.pyplot as plt

family_colors = {

    # Gemma (orange family)
    "gemma12": "#EDC948",   # light yellow-orange
    "gemma27": "#E69F00",   # dark orange

    # Qwen (green family)
    "qwen8":   "#8DD3C7",   # light bluish green
    "qwen14":  "#009E73",   # dark bluish green

    # Ministral (purple family)
    "ministral8":  "#D7B5D8",  # light purple
    "ministral14": "#CC79A7",   # dark purple

    # LLaMA (blue family)
    "llama8":  "#56B4E9"   # light blue
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