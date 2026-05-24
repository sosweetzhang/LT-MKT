import json
from pathlib import Path

import torch


class LTMKTDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        return (
            torch.LongTensor(sequence["concept_ids"]),
            torch.FloatTensor(sequence["responses"]),
            torch.LongTensor(sequence["question_ids"]),
            torch.LongTensor(sequence["question_difficulty"]),
            torch.LongTensor(sequence["domain_transition"]),
            torch.LongTensor(sequence["domain_coverage"]),
        )


def load_dataset_registry(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_dataset_config(root, name):
    registry_path = Path(root) / "data" / "datasets.json"
    registry = load_dataset_registry(registry_path)
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown dataset {name}. Available datasets: {available}")
    config = registry[name]
    if not config.get("available", False):
        raise FileNotFoundError(f"Dataset {name} is registered but data files are not included.")
    return config


def load_sequences(path, seq_len, question_coverage=None):
    raw = Path(path).read_text(encoding="utf-8").strip().splitlines()
    sequences = []
    for offset in range(0, len(raw), 6):
        block = raw[offset : offset + 6]
        if len(block) < 6:
            continue
        length = int(block[0])
        concept_ids = parse_ints(block[1], length)
        responses = parse_ints(block[2], length)
        question_ids = parse_ints(block[3], length)
        question_difficulty = parse_ints(block[4], length)
        domain_signal = parse_ints(block[5], length)
        for start in range(0, length, seq_len):
            end = min(start + seq_len, length)
            item = {
                "concept_ids": pad(concept_ids[start:end], seq_len, 0),
                "responses": pad(responses[start:end], seq_len, -1),
                "question_ids": pad(question_ids[start:end], seq_len, 0),
                "question_difficulty": pad(question_difficulty[start:end], seq_len, 0),
                "domain_transition": pad(domain_signal[start:end], seq_len, 0),
                "domain_coverage": pad(
                    [question_coverage.get(question_id, 1) if question_coverage else 1 for question_id in question_ids[start:end]],
                    seq_len,
                    0,
                ),
            }
            sequences.append(item)
    return sequences


def parse_ints(line, expected_length):
    values = [int(value) for value in line.split(",") if value != ""]
    return values[:expected_length]


def pad(values, length, value):
    return values + [value] * (length - len(values))


def load_graph(path, n_concepts):
    with open(path, "r", encoding="utf-8") as file:
        graph = json.load(file)
    return {
        "intra": build_adjacency(graph.get("successor", {}), n_concepts),
        "inter": build_adjacency(graph.get("similarity", {}), n_concepts),
    }


def build_adjacency(edges, n_concepts):
    adjacency = torch.eye(n_concepts + 1)
    for source, targets in edges.items():
        source_id = int(source)
        if source_id > n_concepts:
            continue
        for target in targets:
            target_id = int(target)
            if 0 < target_id <= n_concepts:
                adjacency[source_id, target_id] = 1.0
    return adjacency


def load_question_concepts(path, n_questions, n_concepts=None):
    with open(path, "r", encoding="utf-8") as file:
        qmatrix = json.load(file)
    max_width = max((len(values) for values in qmatrix.values()), default=1)
    concepts = torch.zeros(n_questions + 1, max_width, dtype=torch.long)
    for question, values in qmatrix.items():
        question_id = int(question)
        if question_id > n_questions:
            continue
        filtered = [int(value) for value in values if int(value) > 0 and (n_concepts is None or int(value) <= n_concepts)]
        for index, concept_id in enumerate(filtered[:max_width]):
            concepts[question_id, index] = concept_id
    return concepts


def load_question_coverage(path, n_questions):
    with open(path, "r", encoding="utf-8") as file:
        qmatrix = json.load(file)
    coverage = {}
    for question in range(1, n_questions + 1):
        values = qmatrix.get(str(question), [])
        coverage[question] = max(1, len([value for value in values if int(value) > 0]))
    return coverage
