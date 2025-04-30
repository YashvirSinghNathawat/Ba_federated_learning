def evaluate_model(model, X, male, y):
    results = model.evaluate([X, male], y, verbose=0)
    return {'loss': results[0], 'mae': results[1]}