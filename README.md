# Bystander Effect in LLMs – Master’s Thesis Repository

This repository contains all code, experimental implementations, and analysis files for the master’s thesis:

**“From Social Psychology to Artificial Intelligence: Investigation of the Bystander Effect in Large Language Models”**

The project investigates whether large language models (LLMs) exhibit bystander-like behavior in multi-entity decision-making settings, inspired by classical findings in social psychology. Specifically, it examines whether the likelihood of an LLM taking action decreases as the number of potential alternative responders increases.

## Repository Structure

The repository is organized into the following components:

### `Email_experiments/`
Contains all experiments testing the bystander effect in an email-based setting. This includes the Yes–No design, choice-based tasks, and related prompts and analyses.  
See the folder-specific README for detailed information.

### `Shared_task_experiments/`
Contains experiments in a multi-agent / shared-task setting, where LLMs operate in collaborative environments. This extends the analysis beyond email-based interaction.  
See the folder-specific README for detailed information.

### `Empathy_analysis/`
Contains additional analyses related to model-level differences on an empathy questionnaire (e.g., empathy-related measures) which relation to bystander-like behavior is investigated.

### `Comparison_Bystander_effect.ipynb`
Jupyter notebook used to aggregate and compare results across all experimental settings. It produces cross-experiment analyses of the bystander effect across models, conditions, and tasks. 
Additional explorations of relationships with other measures, such as benchmark performances, are performed.

### `plots_comparison/`
Contains figures and visualizations summarizing and comparing the magnitude of the bystander effect across models with external variables, such as benchmark performances.

## Project Overview

The project systematically investigates whether LLMs exhibit a **bystander effect**, defined as a reduced likelihood of helping when multiple potential entities are present.

The experimental framework includes:
- Controlled email-based decision tasks (Yes–No and choice settings)
- A shared-task / multi-agent environment
- Manipulations of urgency, team composition and psychological processes
- Comparisons across multiple model families and parameter sizes

The analysis further examines:
- Whether the effect generalizes across different model architectures
- How it varies with model scale and performance
- Whether psychologically motivated mechanisms (e.g., responsibility diffusion) influence model behavior
- How the effect relates to model characteristics, such as general model capability

## Key Outputs

- Cross-experiment comparison of bystander effects across LLMs
- Visualizations of effect sizes and behavioral patterns
- Benchmark-based evaluation of model performance for context
- Aggregated analysis of robustness across experimental conditions

## Notes

- Each experiment folder contains its own detailed README with setup and implementation details.
- All models were run locally using vLLM to ensure controlled and reproducible inference conditions.
- Results should be interpreted as behavioral patterns in LLMs, not direct evidence of human psychological mechanisms.
