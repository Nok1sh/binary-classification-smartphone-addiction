import optuna
from sklearn.metrics import f1_score


def tune_model(model_class, param_space, X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        params = {}

        for param, space in param_space.items():
            if isinstance(space, tuple):
                low, high = space[0], space[1]
                log = (
                    space[2] if len(space) > 2 and isinstance(space[2], bool) else False
                )

                if isinstance(low, int) and isinstance(high, int):
                    params[param] = trial.suggest_int(param, low, high, log=log)
                else:
                    params[param] = trial.suggest_float(param, low, high, log=log)

            elif isinstance(space, list):
                params[param] = trial.suggest_categorical(param, space)

            else:
                params[param] = space

        model = model_class(**params)
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        except TypeError:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        # accuracy = (y_pred == y_val).mean()
        f1 = f1_score(y_val, y_pred, average="weighted")
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    return study.best_params, study.best_value
