import xml.etree.ElementTree as ET
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import re
import torch
import json
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
from collections import defaultdict

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load synthetic texts and labels (3358 samples)
    texts = []
    for i in range(3358):
        with open(f'../data/synthetic/retrieval_reasoning_reports/synthetic_clinical_report_{i}.txt', 'r') as file:
            text = file.read()
            text = re.sub(r'"overall_status":\s*".*?",\s*', '', text)
            text = re.sub(r'"why_stopped":\s*".*?",\s*', '', text)
            text = re.sub(r'failed', '', text, flags=re.IGNORECASE)
            text = re.sub(r'successful', '', text, flags=re.IGNORECASE)
            texts.append(text)

    with open('../data/synthetic/retrieval_reasoning_label.txt', 'r') as file:
        label = file.read()

    labels = [int(x) for x in label.splitlines()]

    synthetic_data = pd.DataFrame({'text': texts, 'label': labels})
    synthetic_dataset = Dataset.from_pandas(synthetic_data)

    # Define helper functions
    def element_to_dict(el):
        children = list(el)
        if not children:
            return el.text
        result = {}
        for child in children:
            child_dict = element_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict
        return result

    def xml_to_dict(element):
        return {element.tag: element_to_dict(element)}

    def read_xml_file(file_path):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return xml_to_dict(root)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return {}

    # Load real data
    file_path = '../data/IQVIA/filtered_trial_outcomes_with_labels.csv'
    df = pd.read_csv(file_path)

    def extract_intervention_name(study_id):
        xml_path = f'../data/trials/{study_id[:7]}xxxx/{study_id}.xml'
        xml_dict = read_xml_file(xml_path)
        if not xml_dict:
            return None
        interventions = xml_dict.get('clinical_study', {}).get('intervention', {})
        if isinstance(interventions, list):
            interventions = interventions[0]  # Assuming only one intervention per study
        intervention_name = interventions.get('intervention_name', '').lower()
        return intervention_name if intervention_name else None

    df['intervention_name'] = df['studyid'].apply(extract_intervention_name)
    df = df.dropna(subset=['intervention_name']).reset_index(drop=True)

    with open('../data/synthetic/generated_intervention_list.txt', 'r') as file:
        intervention_list = [line.strip().lower() for line in file.readlines()]

    intervention_list = list(dict.fromkeys(intervention_list))

    # Filter the dataframe to include only the interventions in the list
    df = df[df['intervention_name'].isin(intervention_list)].reset_index(drop=True)

    # Load examples from real data
    def load_examples(df):
        texts = []
        labels = []
        for idx, row in df.iterrows():
            study_id = row['studyid']
            label = row['label']
            xml_path = f'../data/trials/{study_id[:7]}xxxx/{study_id}.xml'
            xml_dict = read_xml_file(xml_path)
            if not xml_dict:
                continue  # Skip if XML file is missing
            xml_string = json.dumps(xml_dict)
            xml_string = re.sub(r'"overall_status":\s*".*?",\s*', '', xml_string)
            xml_string = re.sub(r'"why_stopped":\s*".*?",\s*', '', xml_string)
            texts.append(xml_string)
            labels.append(label)
        return texts, labels

    real_texts, real_labels = load_examples(df)

    # Print the length of the real dataset
    print(f"Number of real data samples: {len(real_texts)}")

    # Check if there are enough real data samples
    if len(real_texts) < 3358 + 2:  # At least one sample for validation and test
        print("Not enough real data samples.")
        return

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../models/biobert-base-cased-v1.1')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    # Prepare to collect metrics
    metrics_results = {
        '100% synthetic': defaultdict(list),
        '80% synthetic + 20% real': defaultdict(list),
        '60% synthetic + 40% real': defaultdict(list),
        '40% synthetic + 60% real': defaultdict(list),
        '20% synthetic + 80% real': defaultdict(list),
        '100% real': defaultdict(list),
    }

    seed_values = [40, 41, 42]

    for seed in seed_values:
        print(f"\nRunning with seed: {seed}")

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Create DataFrame from real data
        real_data = pd.DataFrame({'text': real_texts, 'label': real_labels})

        # Split real data into training, validation, and test sets using random splitting with seed
        total_real_samples = len(real_data)
        train_size = 3358  # Number of samples you want in your training set
        train_proportion = train_size / total_real_samples

        real_train_data, real_remaining_data = train_test_split(
            real_data,
            train_size=train_proportion,
            random_state=seed,
            stratify=real_data['label']
        )

        real_val_data, real_test_data = train_test_split(
            real_remaining_data,
            test_size=0.5,
            random_state=seed,
            stratify=real_remaining_data['label']
        )

        # Reset indices
        real_train_data = real_train_data.reset_index(drop=True)
        real_val_data = real_val_data.reset_index(drop=True)
        real_test_data = real_test_data.reset_index(drop=True)

        # Tokenize validation and test datasets
        val_dataset = Dataset.from_pandas(real_val_data)
        test_dataset = Dataset.from_pandas(real_test_data)

        tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Prepare training datasets with different compositions
        proportions = {
            '100% synthetic': (1.0, 0.0),
            '80% synthetic + 20% real': (0.8, 0.2),
            '60% synthetic + 40% real': (0.6, 0.4),
            '40% synthetic + 60% real': (0.4, 0.6),
            '20% synthetic + 80% real': (0.2, 0.8),
            '100% real': (0.0, 1.0),
        }

        for comp_name, (syn_ratio, real_ratio) in proportions.items():
            print(f"\nTraining with {comp_name} dataset")

            # Calculate number of samples from synthetic and real data
            syn_samples = int(round(syn_ratio * 3358))
            real_samples = 3358 - syn_samples

            # Sample from synthetic data
            if syn_samples > 0:
                if syn_samples < len(synthetic_data):
                    synthetic_samples, _ = train_test_split(
                        synthetic_data,
                        train_size=syn_samples,
                        random_state=seed,
                        stratify=synthetic_data['label'] if len(synthetic_data['label'].unique()) > 1 else None
                    )
                else:
                    synthetic_samples = synthetic_data.copy()
                synthetic_samples = synthetic_samples.reset_index(drop=True)
                tokenized_synthetic_samples = Dataset.from_pandas(synthetic_samples).map(tokenize_function, batched=True)
            else:
                tokenized_synthetic_samples = None

            # Sample from real training data
            if real_samples > 0:
                if real_samples < len(real_train_data):
                    real_train_samples, _ = train_test_split(
                        real_train_data,
                        train_size=real_samples,
                        random_state=seed,
                        stratify=real_train_data['label']
                    )
                else:
                    real_train_samples = real_train_data.copy()
                real_train_samples = real_train_samples.reset_index(drop=True)
                tokenized_real_train_samples = Dataset.from_pandas(real_train_samples).map(tokenize_function, batched=True)
            else:
                tokenized_real_train_samples = None

            # Combine datasets
            if syn_samples > 0 and real_samples > 0:
                train_dataset = concatenate_datasets([tokenized_synthetic_samples, tokenized_real_train_samples])
            elif syn_samples > 0:
                train_dataset = tokenized_synthetic_samples
            else:
                train_dataset = tokenized_real_train_samples

            # Shuffle the training dataset
            train_dataset = train_dataset.shuffle(seed=seed)

            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(
                '../models/biobert-base-cased-v1.1',
                num_labels=2
            )
            model.to(device)

            training_args = TrainingArguments(
                output_dir=f'./results_{comp_name.replace(" ", "_")}_{seed}',
                eval_strategy="epoch",
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=7,
                weight_decay=0.01,
                logging_dir=f'./logs_{comp_name.replace(" ", "_")}_{seed}',
                logging_steps=10,
                save_strategy="epoch",
                load_best_model_at_end=True,
                learning_rate=1e-5,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                lr_scheduler_type="linear",
                seed=seed
            )

            def compute_metrics(p):
                preds = np.argmax(p.predictions, axis=1)
                labels = p.label_ids
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, average='binary', zero_division=0
                )
                acc = accuracy_score(labels, preds)
                scores = p.predictions[:, 1]
                roc_auc = roc_auc_score(labels, scores)
                pr_auc = average_precision_score(labels, scores)
                return {
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc
                }

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=tokenized_val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            trainer.train()

            # Evaluation on test set
            model.eval()
            tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8)

            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    # Apply softmax to get probabilities
                    probs = torch.softmax(logits, dim=1)
                    probs = probs[:, 1]  # Probability of the positive class

                    # Get predicted classes
                    preds = torch.argmax(logits, dim=1)

                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Convert lists to numpy arrays
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            # Compute evaluation metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )
            roc_auc = roc_auc_score(all_labels, all_probs)
            pr_auc = average_precision_score(all_labels, all_probs)

            print(f"\nPerformance of {comp_name} Model on Test Set (Seed {seed}):")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print(f"ROC AUC:   {roc_auc:.4f}")
            print(f"PR AUC:    {pr_auc:.4f}")

            # Store metrics
            metrics_results[comp_name]['accuracy'].append(accuracy)
            metrics_results[comp_name]['precision'].append(precision)
            metrics_results[comp_name]['recall'].append(recall)
            metrics_results[comp_name]['f1'].append(f1)
            metrics_results[comp_name]['roc_auc'].append(roc_auc)
            metrics_results[comp_name]['pr_auc'].append(pr_auc)

    # After all runs, compute mean and variance
    for comp_name in metrics_results.keys():
        print(f"\nFinal Results for {comp_name}:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            values = metrics_results[comp_name][metric]
            mean = np.mean(values)
            variance = np.var(values)
            print(f"{metric.capitalize()} - Mean: {mean:.4f}, Variance: {variance:.6f}")

if __name__ == "__main__":
    main()
