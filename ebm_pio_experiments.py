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
