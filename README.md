# Retrieval-Reasoning Large Language Model-based Synthetic Clinical Trial Generation

This repository hosts a novel method of utilizing LLM to generate synthetic clinical trials. The general pipeline is described by the figure below:

![main_figure](https://github.com/user-attachments/assets/d0008236-a7e5-4792-8110-2a0ab75bb34a)

## Instrallation

We build conda environment and uses pip to install the required packages as follows:
```bash
conda create -n synct python==3.12
conda activate synct
pip install pandas datasets transformers torch scikit-learn numpy matplotlib seaborn openai tiktoken
```

## Preprocessing

For preprocessing before running the code, please follow these steps:

### ClinicalTrial.gov
```bash
cd data/raw
wget https://clinicaltrials.gov/AllPublicXML.zip
unzip AllPublicXML.zip -d ../trials
```
Run ```bash preprocess/unzip.ipynb``` and ```bashpreprocess/get_trials.py```

### DrugBank
Please download the drugbank ```bash vocabulary.csv``` from [DrugBank](https://go.drugbank.com/releases/latest#open-data) and place it in ```bash data/raw```, then run ```bash preprocess/data_filtering.ipynb```

## Generation

For clinical trial generation, replace the ```bash 'ADD-YOUR-API-KEY'``` in ```bash generation/retrieval_reasoning.ipynb```
You can also see the 3,358 synthetic trials under ```bash data/synthetic/retrieval_reasoning_reports```, within ```bash data/synthetic```, you can also find the intervention names of that the synthetic clinical trials intend to have and they actually have, in ```bash correct_intervention_list.txt``` and ```bash generated_intervention_list.txt``` respectively. The labels of those trials can be found in ```bash retrieval_reasoning_label.txt```

## Experiment

For the in-distribution, generalization and ratio test mentioned in the paper, use the command ```bash cd evaluation```, and run ```bash fine_tune_rr_biobert.py```, ```bash fine_tune_rr_biobert_general.py``` and ```bash ratio.py``` respectively. For visualizations, please refer to ```bash visualization.ipynb```
