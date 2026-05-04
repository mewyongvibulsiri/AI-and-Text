
from __future__ import annotations

import argparse
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ENTITY_TYPES = ("participants", "interventions", "outcomes")
BIO_LABELS = ("O", "B", "I")
DETECTION_LABELS = ("O", "E")
PUNCT_RE = re.compile(r"^\W+$")


@dataclass
class Document:
    doc_id: str
    tokens: List[str]
    tags: List[str]


def get_data_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    script_dir = Path(__file__).resolve().parent
    if (script_dir / "documents").exists() and (script_dir / "annotations").exists():
        return script_dir

    cwd = Path.cwd()
    if (cwd / "documents").exists() and (cwd / "annotations").exists():
        return cwd
    if (cwd / "ebm_nlp_2_00").exists():
        return (cwd / "ebm_nlp_2_00").resolve()
    raise FileNotFoundError(
        "Could not locate the dataset. Pass --data-dir /path/to/ebm_nlp_2_00."
    )


def get_doc_ids(data_dir: Path, split: str, entity_type: str) -> List[str]:
    split_dir = "test/gold" if split == "test" else "train"
    ann_dir = (
        data_dir
        / "annotations"
        / "aggregated"
        / "hierarchical_labels"
        / entity_type
        / split_dir
    )
    return sorted(p.stem.split(".")[0] for p in ann_dir.glob("*.AGGREGATED.ann"))


def load_tokens(data_dir: Path, doc_id: str) -> List[str]:
    path = data_dir / "documents" / f"{doc_id}.tokens"
    return path.read_text(encoding="utf-8").splitlines()


def load_hierarchical_labels(
    data_dir: Path, doc_id: str, entity_type: str, split: str
) -> List[int]:
    split_dir = "test/gold" if split == "test" else "train"
    path = (
        data_dir
        / "annotations"
        / "aggregated"
        / "hierarchical_labels"
        / entity_type
        / split_dir
        / f"{doc_id}.AGGREGATED.ann"
    )
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines()]


def hierarchical_to_bio(tags: Sequence[int]) -> List[str]:
    bio = []
    prev = 0
    for tag in tags:
        if tag == 0:
            bio.append("O")
        elif prev == 0:
            bio.append("B")
        else:
            bio.append("I")
        prev = tag
    return bio


def load_dataset(data_dir: Path, entity_type: str, split: str) -> List[Document]:
    dataset = []
    skipped = 0
    for doc_id in get_doc_ids(data_dir, split, entity_type):
        try:
            tokens = load_tokens(data_dir, doc_id)
            tags = hierarchical_to_bio(
                load_hierarchical_labels(data_dir, doc_id, entity_type, split)
            )
        except FileNotFoundError:
            skipped += 1
            continue
        if len(tokens) != len(tags):
            raise ValueError(
                f"Length mismatch for {entity_type}/{split}/{doc_id}: "
                f"{len(tokens)} tokens vs {len(tags)} labels"
            )
        dataset.append(Document(doc_id=doc_id, tokens=tokens, tags=tags))
    if skipped:
        print(
            f"Skipped {skipped} missing files while loading {entity_type}/{split}.",
            flush=True,
        )
    return dataset


def shape_of(token: str) -> str:
    pieces = []
    for ch in token[:12]:
        if ch.isupper():
            pieces.append("X")
        elif ch.islower():
            pieces.append("x")
        elif ch.isdigit():
            pieces.append("d")
        else:
            pieces.append(ch)
    return "".join(pieces)


def is_punct(token: str) -> bool:
    return bool(PUNCT_RE.match(token))


def token_features(tokens: Sequence[str], i: int, prev_pred: str = "<START>") -> Dict[str, float]:
    tok = tokens[i]
    lower = tok.lower()
    prev_tok = tokens[i - 1] if i > 0 else "<START>"
    next_tok = tokens[i + 1] if i + 1 < len(tokens) else "<END>"
    prev_lower = prev_tok.lower() if i > 0 else "<START>"
    next_lower = next_tok.lower() if i + 1 < len(tokens) else "<END>"

    feats: Dict[str, float] = {
        "bias": 1.0,
        f"tok={lower}": 1.0,
        f"shape={shape_of(tok)}": 1.0,
        f"prev={prev_lower}": 1.0,
        f"next={next_lower}": 1.0,
        f"bigram_prev={prev_lower}|{lower}": 1.0,
        f"bigram_next={lower}|{next_lower}": 1.0,
        f"prev_pred={prev_pred}": 1.0,
        f"istitle={tok.istitle()}": 1.0,
        f"isupper={tok.isupper()}": 1.0,
        f"isdigit={tok.isdigit()}": 1.0,
        f"hasdigit={any(ch.isdigit() for ch in tok)}": 1.0,
        f"hyphen={'-' in tok}": 1.0,
        f"punct={is_punct(tok)}": 1.0,
        f"prefix1={lower[:1]}": 1.0,
        f"prefix2={lower[:2]}": 1.0,
        f"prefix3={lower[:3]}": 1.0,
        f"suffix1={lower[-1:]}": 1.0,
        f"suffix2={lower[-2:]}": 1.0,
        f"suffix3={lower[-3:]}": 1.0,
    }
    return feats


class AveragedPerceptron:
    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.weights: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.totals: Dict[Tuple[str, str], float] = defaultdict(float)
        self.timestamps: Dict[Tuple[str, str], int] = defaultdict(int)
        self.step = 0

    def score(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {label: 0.0 for label in self.labels}
        for feat, value in features.items():
            weights = self.weights.get(feat)
            if not weights:
                continue
            for label, weight in weights.items():
                scores[label] += weight * value
        return scores

    def predict(self, features: Dict[str, float]) -> str:
        scores = self.score(features)
        return max(self.labels, key=lambda label: (scores[label], label))

    def update(self, truth: str, guess: str, features: Dict[str, float]) -> None:
        self.step += 1
        if truth == guess:
            return
        for feat, value in features.items():
            self._update_feat(feat, truth, value)
            self._update_feat(feat, guess, -value)

    def _update_feat(self, feat: str, label: str, delta: float) -> None:
        param = (feat, label)
        current = self.weights[feat].get(label, 0.0)
        elapsed = self.step - self.timestamps[param]
        self.totals[param] += elapsed * current
        self.timestamps[param] = self.step
        self.weights[feat][label] = current + delta

    def average_weights(self) -> None:
        for feat, label_weights in list(self.weights.items()):
            new_weights = {}
            for label, weight in list(label_weights.items()):
                param = (feat, label)
                elapsed = self.step - self.timestamps[param]
                total = self.totals[param] + elapsed * weight
                averaged = total / max(self.step, 1)
                if averaged:
                    new_weights[label] = averaged
            if new_weights:
                self.weights[feat] = new_weights
            else:
                del self.weights[feat]


class SequenceTagger:
    def __init__(self, labels: Sequence[str], epochs: int = 4, seed: int = 13) -> None:
        self.labels = list(labels)
        self.epochs = epochs
        self.seed = seed
        self.model = AveragedPerceptron(labels)

    def train(self, documents: Sequence[Document]) -> "SequenceTagger":
        docs = list(documents)
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            rng.shuffle(docs)
            for doc in docs:
                prev_pred = "<START>"
                for i, gold in enumerate(doc.tags):
                    feats = token_features(doc.tokens, i, prev_pred=prev_pred)
                    guess = self.model.predict(feats)
                    self.model.update(gold, guess, feats)
                    prev_pred = guess
        self.model.average_weights()
        return self

    def predict(self, tokens: Sequence[str]) -> List[str]:
        preds = []
        prev_pred = "<START>"
        for i in range(len(tokens)):
            feats = token_features(tokens, i, prev_pred=prev_pred)
            pred = self.model.predict(feats)
            preds.append(pred)
            prev_pred = pred
        return preds


class DecomposedTagger:
    def __init__(self, epochs: int = 4, seed: int = 13) -> None:
        self.detector = SequenceTagger(DETECTION_LABELS, epochs=epochs, seed=seed)

    @staticmethod
    def to_detection_docs(documents: Sequence[Document]) -> List[Document]:
        converted = []
        for doc in documents:
            tags = ["E" if tag != "O" else "O" for tag in doc.tags]
            converted.append(Document(doc_id=doc.doc_id, tokens=doc.tokens, tags=tags))
        return converted

    def train(self, documents: Sequence[Document]) -> "DecomposedTagger":
        self.detector.train(self.to_detection_docs(documents))
        return self

    def predict(self, tokens: Sequence[str]) -> List[str]:
        det = self.detector.predict(tokens)
        bio = []
        prev_e = False
        for label in det:
            if label == "O":
                bio.append("O")
                prev_e = False
            else:
                bio.append("I" if prev_e else "B")
                prev_e = True
        return bio


ZERO_SHOT_CUES = {
    "participants": {
        "patient", "patients", "participant", "participants", "subject", "subjects",
        "adult", "adults", "child", "children", "infant", "infants", "women",
        "woman", "men", "man", "neonate", "neonates", "volunteer", "volunteers",
        "pregnant", "elderly", "healthy", "smokers", "diabetic", "hypertensive",
    },
    "interventions": {
        "treatment", "treated", "therapy", "therapies", "intervention", "drug",
        "drugs", "dose", "doses", "placebo", "surgery", "surgical", "exercise",
        "diet", "vaccine", "vaccination", "insulin", "aspirin", "metformin",
        "chemotherapy", "radiotherapy", "supplement", "tablet", "capsule", "mg",
    },
    "outcomes": {
        "outcome", "outcomes", "result", "results", "response", "responses",
        "mortality", "survival", "score", "scores", "pain", "efficacy", "safety",
        "blood", "pressure", "glucose", "cholesterol", "adverse", "event",
        "events", "quality", "remission", "improvement", "improved", "reduction",
    },
}

LEFT_BOUNDARY_WORDS = {"the", "a", "an", "of", "with", "and", "or", "to", "in", "for"}


class ZeroShotHeuristicTagger:
    def __init__(self, entity_type: str) -> None:
        self.entity_type = entity_type
        self.cues = ZERO_SHOT_CUES[entity_type]

    def predict(self, tokens: Sequence[str]) -> List[str]:
        marks = [False] * len(tokens)
        lowers = [tok.lower() for tok in tokens]
        for i, lower in enumerate(lowers):
            if lower not in self.cues:
                continue
            start = i
            end = i
            while start > 0:
                prev = lowers[start - 1]
                if is_punct(tokens[start - 1]) or prev in LEFT_BOUNDARY_WORDS:
                    break
                if len(tokens[start - 1]) <= 2 and not tokens[start - 1].isdigit():
                    break
                start -= 1
            while end + 1 < len(tokens):
                nxt = lowers[end + 1]
                if is_punct(tokens[end + 1]) or nxt in {"and", "or", "but"}:
                    break
                if len(tokens[end + 1]) <= 1:
                    break
                end += 1
                if end - i >= 4:
                    break
            for j in range(start, end + 1):
                marks[j] = True

        bio = []
        prev = False
        for marked in marks:
            if not marked:
                bio.append("O")
                prev = False
            else:
                bio.append("I" if prev else "B")
                prev = True
        return bio


def spans_from_tags(tags: Sequence[str]) -> List[Tuple[int, int]]:
    spans = []
    start = None
    for i, tag in enumerate(tags):
        if tag == "B":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == "O":
            if start is not None:
                spans.append((start, i - 1))
                start = None
        elif tag == "I":
            if start is None:
                start = i
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans


def evaluate_documents(
    gold_docs: Sequence[Document], pred_tags_by_doc: Dict[str, List[str]]
) -> Dict[str, float]:
    tp = fp = fn = 0
    token_correct = 0
    token_total = 0
    gold_nonempty_docs = 0
    pred_nonempty_docs = 0
    docs_with_any_correct_span = 0
    exact_span_match_docs = 0
    for doc in gold_docs:
        pred = pred_tags_by_doc[doc.doc_id]
        gold_spans = set(spans_from_tags(doc.tags))
        pred_spans = set(spans_from_tags(pred))
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
        token_correct += sum(1 for g, p in zip(doc.tags, pred) if g == p)
        token_total += len(doc.tags)
        if gold_spans:
            gold_nonempty_docs += 1
        if pred_spans:
            pred_nonempty_docs += 1
        if gold_spans & pred_spans:
            docs_with_any_correct_span += 1
        if gold_spans == pred_spans:
            exact_span_match_docs += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = token_correct / token_total if token_total else 0.0
    extraction_coverage = pred_nonempty_docs / len(gold_docs) if gold_docs else 0.0
    usable_doc_rate = docs_with_any_correct_span / len(gold_docs) if gold_docs else 0.0
    exact_doc_match_rate = exact_span_match_docs / len(gold_docs) if gold_docs else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "token_accuracy": accuracy,
        "coverage": extraction_coverage,
        "usable_doc_rate": usable_doc_rate,
        "exact_doc_match_rate": exact_doc_match_rate,
        "gold_docs_with_entity": float(gold_nonempty_docs),
        "pred_docs_with_entity": float(pred_nonempty_docs),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def sample_few_shot(train_docs: Sequence[Document], shots: int, seed: int) -> List[Document]:
    if shots >= len(train_docs):
        return list(train_docs)
    rng = random.Random(seed)
    docs = list(train_docs)
    rng.shuffle(docs)
    return docs[:shots]


def run_experiment(
    entity_type: str,
    train_docs: Sequence[Document],
    test_docs: Sequence[Document],
    formulation: str,
    strategy: str,
    few_shot_docs: int,
    seed: int,
    epochs: int,
) -> Dict[str, object]:
    if strategy == "zero-shot":
        model = ZeroShotHeuristicTagger(entity_type)
        used_train_docs = 0
    else:
        subset = (
            sample_few_shot(train_docs, few_shot_docs, seed)
            if strategy == "few-shot"
            else list(train_docs)
        )
        used_train_docs = len(subset)
        if formulation == "end-to-end":
            model = SequenceTagger(BIO_LABELS, epochs=epochs, seed=seed).train(subset)
        else:
            model = DecomposedTagger(epochs=epochs, seed=seed).train(subset)

    pred_tags = {doc.doc_id: model.predict(doc.tokens) for doc in test_docs}
    metrics = evaluate_documents(test_docs, pred_tags)
    result: Dict[str, object] = {
        "entity_type": entity_type,
        "formulation": formulation,
        "strategy": strategy,
        "train_docs_used": used_train_docs,
        **metrics,
    }
    return result


def print_table(rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "entity_type",
        "formulation",
        "strategy",
        "train_docs_used",
        "precision",
        "recall",
        "f1",
        "token_accuracy",
        "coverage",
        "usable_doc_rate",
    ]
    widths = {h: len(h) for h in headers}
    formatted = []
    for row in rows:
        out = {}
        for h in headers:
            value = row[h]
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            out[h] = text
            widths[h] = max(widths[h], len(text))
        formatted.append(out)

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "-+-".join("-" * widths[h] for h in headers)
    print(header_line, flush=True)
    print(sep_line, flush=True)
    for row in formatted:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers), flush=True)


def summarize_results(rows: Sequence[Dict[str, object]]) -> None:
    print("\nRanked by span F1:", flush=True)
    ranked = sorted(rows, key=lambda row: row["f1"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        print(
            f"{idx:>2}. {row['entity_type']:<13} "
            f"{row['formulation']:<11} {row['strategy']:<9} "
            f"F1={row['f1']:.4f} P={row['precision']:.4f} R={row['recall']:.4f}"
        , flush=True)

    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["formulation"], row["strategy"])].append(float(row["f1"]))

    print("\nAverage F1 across entity types:", flush=True)
    averages = []
    for (formulation, strategy), values in grouped.items():
        avg_f1 = sum(values) / len(values)
        averages.append((avg_f1, formulation, strategy))
    for avg_f1, formulation, strategy in sorted(averages, reverse=True):
        print(f"- {formulation:<11} {strategy:<9} avg_f1={avg_f1:.4f}", flush=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-contained EBM-NLP experiments.")
    parser.add_argument("--data-dir", default=None, help="Path to ebm_nlp_2_00")
    parser.add_argument("--few-shot-docs", type=int, default=40, help="Number of train docs for few-shot runs")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs for perceptron models")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--entity-types",
        nargs="+",
        default=list(ENTITY_TYPES),
        choices=list(ENTITY_TYPES),
        help="Entity types to evaluate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    data_dir = get_data_dir(args.data_dir)
    print(f"Using dataset: {data_dir}", flush=True)
    print(
        "Comparing end-to-end BIO tagging against a decomposed detection->BIO pipeline "
        f"with zero-shot, few-shot ({args.few_shot_docs} docs), and full-train settings.\n"
    , flush=True)

    all_results = []
    for entity_type in args.entity_types:
        train_docs = load_dataset(data_dir, entity_type, "train")
        test_docs = load_dataset(data_dir, entity_type, "test")
        print(
            f"[{entity_type}] train_docs={len(train_docs)} "
            f"test_docs={len(test_docs)}"
        , flush=True)
        for formulation in ("end-to-end", "decomposed"):
            for strategy in ("zero-shot", "few-shot", "full-train"):
                print(f"  running {formulation:11} + {strategy:9} ...", flush=True)
                result = run_experiment(
                    entity_type=entity_type,
                    train_docs=train_docs,
                    test_docs=test_docs,
                    formulation=formulation,
                    strategy=strategy,
                    few_shot_docs=args.few_shot_docs,
                    seed=args.seed,
                    epochs=args.epochs,
                )
                all_results.append(result)

    print("\nDetailed results:", flush=True)
    print_table(all_results)
    summarize_results(all_results)
    best = max(all_results, key=lambda row: row["f1"])
    print(
        "\nBest configuration: "
        f"{best['entity_type']} | {best['formulation']} | {best['strategy']} "
        f"| F1={best['f1']:.4f}"
    , flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

class SequenceTagger:
    def __init__(self, labels: Sequence[str], epochs: int = 4, seed: int = 13) -> None:
        self.labels = list(labels)
        self.epochs = epochs
        self.seed = seed
        self.model = AveragedPerceptron(labels)

    def train(self, documents: Sequence[Document]) -> "SequenceTagger":
        docs = list(documents)
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            rng.shuffle(docs)
            for doc in docs:
                prev_pred = "<START>"
                for i, gold in enumerate(doc.tags):
                    feats = token_features(doc.tokens, i, prev_pred=prev_pred)
                    guess = self.model.predict(feats)
                    self.model.update(gold, guess, feats)
                    prev_pred = guess
        self.model.average_weights()
        return self

    def predict(self, tokens: Sequence[str]) -> List[str]:
        preds = []
        prev_pred = "<START>"
        for i in range(len(tokens)):
            feats = token_features(tokens, i, prev_pred=prev_pred)
            pred = self.model.predict(feats)
            preds.append(pred)
            prev_pred = pred
        return preds

class DecomposedTagger:
    def __init__(self, epochs: int = 4, seed: int = 13) -> None:
        self.detector = SequenceTagger(DETECTION_LABELS, epochs=epochs, seed=seed)

    @staticmethod
    def to_detection_docs(documents: Sequence[Document]) -> List[Document]:
        converted = []
        for doc in documents:
            tags = ["E" if tag != "O" else "O" for tag in doc.tags]
            converted.append(Document(doc_id=doc.doc_id, tokens=doc.tokens, tags=tags))
        return converted

    def train(self, documents: Sequence[Document]) -> "DecomposedTagger":
        self.detector.train(self.to_detection_docs(documents))
        return self

    def predict(self, tokens: Sequence[str]) -> List[str]:
        det = self.detector.predict(tokens)
        bio = []
        prev_e = False
        for label in det:
            if label == "O":
                bio.append("O")
                prev_e = False
            else:
                bio.append("I" if prev_e else "B")
                prev_e = True
        return bio
