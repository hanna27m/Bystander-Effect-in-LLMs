# Investigating the Bystander Effect in Email Experiments

This repository contains simulation code and analysis exploring the **Bystander Effect in Large Language Models (LLMs)** within structured email-based decision scenarios.

The project investigates whether LLM behavior changes depending on the number of recipients and contextual framing, inspired by the classical bystander effect in social psychology.

---

## 📁 Repository Structure

```
.
├── main.py
├── *.py
├── *.ipynb
├── results/
├── plots/
└── README.md
```

### 🔹 `main.py`
Entry point for running all simulations.

- Executes all experiment variants  
- Stores outputs in the `results/` directory  
- Serves as the main script for reproducing experiments  

---

### 🔹 Simulation Files (`*.py`)
Each Python file represents a specific simulation setup or experimental condition.

These scripts:
- Define experimental configurations  
- Generate prompts  
- Run model evaluations  
- Save structured outputs  

---

### 🔹 Notebooks (`*.ipynb`)
Jupyter notebooks used for:

- Statistical analysis   
- Visualization  
- Hypothesis testing  

The notebooks read data from `results/` and generate figures stored in `plots/`.

---

### 🔹 `results/`
Contains outputs from simulations.

This folder ensures reproducibility of the analysis.

---

### 🔹 `plots/`
Stores all figures generated from the notebooks, including:


---


## 🔁 Reproducibility

To fully reproduce results:

1. Clone the repository  
2. Install dependencies  
3. Run `main.py`  
4. Execute the notebooks  

---


*For questions or collaboration inquiries, feel free to open an issue.*
