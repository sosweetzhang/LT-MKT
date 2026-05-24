import torch


class LTMKTTrainer:
    def __init__(self, model, graphs, question_concepts, device):
        self.model = model
        self.graphs = graphs
        self.question_concepts = question_concepts
        self.device = device

    def train_epoch(self, optimizer, loader):
        self.model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            batch = [item.to(self.device) for item in batch]
            loss, count, _, _ = self.step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * count
            total_count += count
        return total_loss / max(total_count, 1)

    def evaluate(self, loader):
        self.model.eval()
        predictions = []
        targets = []
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in loader:
                batch = [item.to(self.device) for item in batch]
                loss, count, pred, target = self.step(batch)
                total_loss += loss.item() * count
                total_count += count
                predictions.extend(pred.detach().cpu().tolist())
                targets.extend(target.detach().cpu().tolist())
        return {
            "loss": total_loss / max(total_count, 1),
            "auc": auc_score(targets, predictions),
            "acc": accuracy_score(targets, predictions),
            "rmse": rmse_score(targets, predictions),
        }

    def step(self, batch):
        concept_ids, responses, question_ids, question_difficulty, domain_transition, domain_coverage = batch
        predictions = self.model(
            concept_ids,
            responses,
            question_ids,
            question_difficulty,
            domain_transition,
            domain_coverage,
            self.graphs["intra"],
            self.graphs["inter"],
            self.question_concepts,
        )
        mask = responses[:, 1:] >= 0
        pred = predictions[:, 1:][mask]
        target = responses[:, 1:][mask]
        loss = torch.nn.functional.binary_cross_entropy(pred, target)
        return loss, int(mask.sum().item()), pred, target


def accuracy_score(targets, predictions):
    if not targets:
        return 0.0
    correct = 0
    for target, prediction in zip(targets, predictions):
        correct += int((prediction >= 0.5) == (target >= 0.5))
    return correct / len(targets)


def rmse_score(targets, predictions):
    if not targets:
        return 0.0
    error = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
    return (error / len(targets)) ** 0.5


def auc_score(targets, predictions):
    pairs = sorted(zip(predictions, targets), key=lambda item: item[0])
    positives = sum(1 for _, target in pairs if target >= 0.5)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2
        for item in range(index, end):
            if pairs[item][1] >= 0.5:
                rank_sum += avg_rank
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
