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

    # Load synthetic texts and labels
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

    data = pd.DataFrame({'text': texts, 'label': labels})
    dataset = Dataset.from_pandas(data)

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
        tree = ET.parse(file_path)
        root = tree.getroot()
        return xml_to_dict(root)

    # Load real data
    file_path = '../data/IQVIA/filtered_trial_outcomes_with_labels.csv'
    df = pd.read_csv(file_path)

    def extract_intervention_name(study_id):
        xml_path = f'../data/trials/{study_id[:7]}xxxx/{study_id}.xml'
        xml_dict = read_xml_file(xml_path)
        xml_string = json.dumps(xml_dict)
        pattern = r'"intervention_name":\s*"([^"]*)"'
        match = re.search(pattern, xml_string)
        if match:
            intervention_name = match.group(1).lower()
            return intervention_name

    df['intervention_name'] = df['studyid'].apply(extract_intervention_name)

    with open('../data/synthetic/generated_intervention_list.txt', 'r') as file:
        intervention_list = [line.strip() for line in file.readlines()]

    intervention_list = list(dict.fromkeys(intervention_list))

    # Filter the dataframe to include only the interventions in the list
    df = df[df['intervention_name'].isin(intervention_list)]

    def load_examples():
        texts = []
        labels = []
        for study_id in df['studyid']:
            label = df.loc[df['studyid'] == study_id, 'label'].values[0]
            xml_path = f'../data/trials/{study_id[:7]}xxxx/{study_id}.xml'
            xml_dict = read_xml_file(xml_path)
            xml_string = json.dumps(xml_dict)
            xml_string = re.sub(r'"overall_status":\s*".*?",\s*', '', xml_string)
            xml_string = re.sub(r'"why_stopped":\s*".*?",\s*', '', xml_string)
            texts.append(xml_string)
            labels.append(label)
        return texts, labels

    real_texts, real_labels = load_examples()

    real_dataset = Dataset.from_dict({'text': real_texts, 'label': real_labels})

    # Create Dataset for synthetic data
    synthetic_dataset = Dataset.from_dict({'text': texts, 'label': labels})

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../models/biobert-base-cased-v1.1')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    # Tokenize datasets
    tokenized_synthetic_dataset = synthetic_dataset.map(tokenize_function, batched=True)
    tokenized_real_dataset = real_dataset.map(tokenize_function, batched=True)

    # Prepare training datasets
    train_datasets = {
        'hybrid': concatenate_datasets([real_dataset, synthetic_dataset]),
        'real': real_dataset,
        'synthetic': synthetic_dataset
    }

    # Tokenize the hybrid dataset
    tokenized_train_dataset_hybrid = train_datasets['hybrid'].map(tokenize_function, batched=True)

    tokenized_train_datasets = {
        'hybrid': tokenized_train_dataset_hybrid,
        'real': tokenized_real_dataset,
        'synthetic': tokenized_synthetic_dataset
    }

    # Prepare to collect metrics
    metrics_results = {
        'no_fine_tuning': defaultdict(list),  # Added for baseline testing
        'hybrid': defaultdict(list),
        'real': defaultdict(list),
        'synthetic': defaultdict(list)
    }

    seed_values = [40, 41, 42]

    for seed in seed_values:
        print(f"\nRunning with seed: {seed}")

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Shuffle the hybrid training dataset
        tokenized_train_datasets['hybrid'] = tokenized_train_datasets['hybrid'].shuffle(seed=seed)

        # Split real data into train, validation, test sets
        real_train_texts, real_temp_texts, real_train_labels, real_temp_labels = train_test_split(
            real_texts,
            real_labels,
            train_size=0.60,
            random_state=seed,
            stratify=real_labels
        )

        # Print the lengths after the first split
        print(f"Training set size: {len(real_train_texts)}")

        real_val_texts, real_test_texts, real_val_labels, real_test_labels = train_test_split(
            real_temp_texts,
            real_temp_labels,
            train_size=0.5,
            random_state=seed,
            stratify=real_temp_labels
        )

        # Print the lengths after the second split
        print(f"Validation set size: {len(real_val_texts)}")
        print(f"Test set size: {len(real_test_texts)}")

        # Training set (real data)
        real_train_dataset = Dataset.from_dict({
            'text': real_train_texts,
            'label': real_train_labels
        })

        # Validation set
        val_dataset = Dataset.from_dict({
            'text': real_val_texts,
            'label': real_val_labels
        })

        # Test set
        test_dataset = Dataset.from_dict({
            'text': real_test_texts,
            'label': real_test_labels
        })

        # Tokenize validation and test datasets
        tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Update the real training dataset for this seed
        tokenized_train_datasets['real'] = real_train_dataset.map(tokenize_function, batched=True)
        # Update the hybrid training dataset (real train + synthetic data)
        train_datasets['hybrid'] = concatenate_datasets([real_train_dataset, synthetic_dataset])
        tokenized_train_datasets['hybrid'] = train_datasets['hybrid'].map(tokenize_function, batched=True)
        tokenized_train_datasets['hybrid'] = tokenized_train_datasets['hybrid'].shuffle(seed=seed)

        # -------------------- Baseline Testing (No Fine-Tuning) --------------------
        print("\nEvaluating pre-trained BioBERT without fine-tuning")

        # Load the pre-trained BioBERT model
        model = AutoModelForSequenceClassification.from_pretrained(
            '../models/biobert-base-cased-v1.1',
            num_labels=2
        )
        model.to(device)

        # Set the format of the test dataset to return PyTorch tensors
        tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        # Create a DataLoader for the test dataset
        test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8)

        # Put the model in evaluation mode
        model.eval()

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

        # Store metrics
        metrics_results['no_fine_tuning']['accuracy'].append(accuracy)
        metrics_results['no_fine_tuning']['precision'].append(precision)
        metrics_results['no_fine_tuning']['recall'].append(recall)
        metrics_results['no_fine_tuning']['f1'].append(f1)
        metrics_results['no_fine_tuning']['roc_auc'].append(roc_auc)
        metrics_results['no_fine_tuning']['pr_auc'].append(pr_auc)

        # -------------------- Training and Evaluation --------------------
        for train_name, train_dataset in tokenized_train_datasets.items():
            print(f"\nTraining with {train_name} dataset")

            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(
                '../models/biobert-base-cased-v1.1',
                num_labels=2
            )
            model.to(device)

            training_args = TrainingArguments(
                output_dir=f'./results_{train_name}_{seed}',
                eval_strategy="epoch",
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=7,
                weight_decay=0.01,
                logging_dir=f'./logs_{train_name}_{seed}',
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

            # Move model to the appropriate device
            model.to(device)

            # Set the format of the test dataset to return PyTorch tensors
            tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            # Create a DataLoader for the test dataset
            test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8)

            # Put the model in evaluation mode
            model.eval()

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

            # Store metrics
            metrics_results[train_name]['accuracy'].append(accuracy)
            metrics_results[train_name]['precision'].append(precision)
            metrics_results[train_name]['recall'].append(recall)
            metrics_results[train_name]['f1'].append(f1)
            metrics_results[train_name]['roc_auc'].append(roc_auc)
            metrics_results[train_name]['pr_auc'].append(pr_auc)

    # After all runs, compute mean and standard deviation
    for train_name in ['no_fine_tuning', 'hybrid', 'real', 'synthetic']:
        print(f"\nFinal Results for {train_name.replace('_', ' ').capitalize()} Dataset:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            values = metrics_results[train_name][metric]
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric.capitalize()} - Mean: {mean:.4f}, Std: {std:.4f}")

if __name__ == "__main__":
    main()

