def run_experiment(
    entity_type,
    train_docs,
    test_docs,
    formulation,
    strategy,
    few_shot_docs,
    seed,
    epochs,
):
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

    return {
        "entity_type": entity_type,
        "formulation": formulation,
        "strategy": strategy,
        "train_docs_used": used_train_docs,
        **metrics,
    }

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
