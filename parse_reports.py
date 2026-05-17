# raw CSV -> parse -> cleanjsonl -> split into train val test
# preprocessing 

import csv
import json
import re

def clean_report(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def clean_impression(text):
    text = re.sub(r'^\s*:\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*+', '', text)  # strip any remaining lone asterisks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n*ATTENDING PHYSICIAN AGREEMENT:.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n*Pulmonary nodule follow-up recommendation:.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

healthy_data = []
with open('resources/og/healthy_chest_ct_synthetic_radiology_reports.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:

        if 'IMPRESSION' not in row[1]:
            continue

        res = row[1].split('IMPRESSION')
        healthy_data.append({
            'diagnosis': 'healthy',
            'report': clean_report(res[0]),
            'impression': clean_impression(res[1])
        })

print(len(healthy_data))

positive_data = []
with open('resources/og/positive_chest_ct_synthetic_radiology_reports.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:

        if 'IMPRESSION' not in row[1]:
            continue

        res = row[1].split('IMPRESSION')
        positive_data.append({
            'diagnosis': row[0],
            'report': clean_report(res[0]),
            'impression': clean_impression(res[1])
        })

print(len(positive_data))

# tvt split

n = len(healthy_data)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
train_healthy = healthy_data[:n_train]
val_healthy = healthy_data[n_train:n_train + n_val]
test_healthy = healthy_data[n_train + n_val:]

# Split by pathology - 70/15/15 per pathology

from collections import defaultdict

by_pathology = defaultdict(list)
for record in positive_data:
    by_pathology[record['diagnosis']].append(record)

train_positive = []
val_positive = []
test_positive = []

for diagnosis, records in by_pathology.items():
    n = len(records)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_positive.extend(records[:n_train])
    val_positive.extend(records[n_train:n_train + n_val])
    test_positive.extend(records[n_train + n_val:])

SYSTEM_PROMPT = (
    "You are a radiologist. Given the clinical indication and CT findings, "
    "write the impression section of the radiology report. "
    "Include: the primary diagnosis or conclusion, key supporting findings, "
    "any clinically significant incidental findings, and appropriate follow-up recommendations. "
    "Be concise and clinically accurate."
)

def to_messages(records):
    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["report"]},
                {"role": "assistant", "content": r["impression"]},
            ]
        }
        for r in records
    ]

# save raw splits
with open('resources/train.jsonl', 'w') as file:
    json.dump(train_healthy + train_positive, file, indent=4)

with open('resources/val.jsonl', 'w') as file:
    json.dump(val_healthy + val_positive, file, indent=4)

with open('resources/test.jsonl', 'w') as file:
    json.dump(test_healthy + test_positive, file, indent=4)

# save formatted splits for SFT
for split, records in [('train', train_healthy + train_positive),
                        ('val', val_healthy + val_positive)]:
    with open(f'resources/sft_{split}.jsonl', 'w') as file:
        for row in to_messages(records):
            file.write(json.dumps(row) + '\n')