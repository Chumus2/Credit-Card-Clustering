import os
import optuna
import joblib
import pandas as pd
from pathlib import Path

import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



def save_model_result(
    results: list,
    model_name: str,
    x: pd.DataFrame,
    model,
    **fit_params
) -> list:
    """For model argument calculating silhouette, davies and calinski score
    results saving to results argument list and returning it.

    Fitting model x argument + extra fit_params. Checking if certain model
    has attribute labels_
    1. if model has labels_:
        - finding model labels (model.labels_)
    2. otherwise:
        - predicting labels (model.predict(x))
    Then calculating n_clusters
    1. if model hasn't at least 2 clusters:
        - printing failure message
        - returning nothing
    Calculating silhouette, davies and calinski score for model.
    Model scores rounding 2, appending to results list argument
    and returning result list.

    function required arguments:
    1. results (python list) list where model results will be saved:
    2. model_name (str) just model name:
    3. x (pd.DataFrame) feature that will be required for model prediction:
    4. model (scikit-model) the trained model itself:
    5. **fit_params (not optional) extra parameters for model prediction:

    function returns:
    1. results (list) list with model results:
    """

    model.fit(x, **fit_params)

    if hasattr(model, 'labels_'):
        labels = model.labels_
    else:
        labels = model.predict(x)

    mask = labels != -1
    n_clusters = len(set(labels[mask]))

    if n_clusters < 2:
        print(f"Failed to save model results for {model_name} because there are only {n_clusters} clusters.")
        return results

    model_silhouette = silhouette_score(x[mask], labels[mask])
    model_davies = davies_bouldin_score(x[mask], labels[mask])
    model_calinski = calinski_harabasz_score(x[mask], labels[mask])

    results.append({
        "model_name": model_name,
        "silhouette_score": round(model_silhouette, 3),
        "davies_score": round(model_davies, 3),
        "calinski_score": round(model_calinski, 3),
    })

    return results


def print_model_result(
    results: list
) -> None:
    """From results list argument printing results beautifully

    Takes results list (argument) and checking
    1. if results empty:
        - printing that model results are empty
        - returns
    Make pd.DataFrame for results list. And printing results.

    function required arguments:
    1. results (list) list where is model results:

    No return.
    """

    if not results:
        print("Model results are empty")
        return

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print(" " * 20 + "Clustering Results")
    print("=" * 65)
    print(result_df.to_string(index=False))
    print("\n"+ "=" * 65)


def create_cluster_objective(
    x: pd.DataFrame,
    algorithm: str = "dbscan",

    # Parameters for dbscan
    eps_min: float = 0.1,
    eps_max: float = 2.0,
    min_samples_min: int = 3,
    min_samples_max: int = 20,
    max_noise_ratio: float = 0.4,

    # Parameters for hdbscan
    min_cluster_size_min: int = 5,
    min_cluster_size_max: int = 50,
):
    """Function for creating objective-function for optuna with df x argument.

    Function checking algorithm argument:
    1. if algorithm is dbscan:
        - suggesting eps and min_samples
        - creating DBSCAN model
    2. if algorithm is hdbscan:
        - suggesting min_cluster_size and min_samples
        - creating HDBSCAN model
    3. otherwise:
        - raising ValueError for invalid algorithm

    Predicting model labels for x.
    Calculating n_clusters (excluding noise -1).
    1. if n_clusters < 2:
        - returning -1.0 penalty score

    Calculating noise_ratio.
    1. if noise_ratio > max_noise_ratio:
        - returning -1.0 penalty score

    Calculating silhouette score for non-noise samples (mask) and returning it.

    function required arguments:
    1. x (pd.DataFrame) feature that will be required for model prediction:
    2. algorithm (str) model algorithm name ('dbscan' or 'hdbscan'):

    function optional arguments:
    1. eps_min (float) minimum eps value for dbscan:
    2. eps_max (float) maximum eps value for dbscan:
    3. min_samples_min (int) minimum min_samples value for models:
    4. min_samples_max (int) maximum min_samples value for models:
    5. max_noise_ratio (float) maximum noise ratio allowed:
    6. min_cluster_size_min (int) minimum min_cluster_size value for hdbscan:
    7. min_cluster_size_max (int) maximum min_cluster_size value for hdbscan:

    function returns:
    1. objective (function) function for optuna study:
    """

    def objective(trial: optuna.Trial) -> float:
        algo = algorithm.lower()

        if algo == "dbscan":
            eps = trial.suggest_float("eps", eps_min, eps_max, step=0.05)
            min_samples = trial.suggest_int("min_samples", min_samples_min, min_samples_max)
            model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )

        elif algo == "hdbscan":
            min_cluster_size = trial.suggest_int("min_cluster_size", min_cluster_size_min, min_cluster_size_max)
            min_samples = trial.suggest_int("min_samples", min_samples_min, min_samples_max)
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                core_dist_n_jobs=-1
            )
        else:
            raise ValueError("Invalid algorithm argument")

        labels = model.fit_predict(x)

        real_clusters = set(labels) - {-1}
        if len(real_clusters) < 2:
            return -1.0

        noise_ratio = (labels == -1).sum() / len(labels)
        if noise_ratio > max_noise_ratio:
            return -1.0

        mask = labels != -1
        return silhouette_score(x[mask], labels[mask])

    return objective


def save_trained_model(
    model,
    output_path: str | Path
) -> None:
    """Save a trained model to path specified by output_path argument using joblib.

    Firstly finding place to save model by output_path argument (pathlib.Path).
    Then trying save model to this path.

    1. if path exists:
        - saves model to it
        - printing success message
        - printing model class info
    2. otherwise:
        - printing failure message
        - raises error

    function required arguments:
    1. model (scikit-learn / hdbscan estimator) model object to save:
    2. output_path (str / pathlib.Path) path to save model:

    No return.
    """

    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not output_path.parent.exists():
            raise FileNotFoundError(f"Target directory does not exist: {output_path.parent}")

        joblib.dump(model, output_path)
        print(f"Successfully saved model: {output_path}")

        model_name = model.__class__.__name__
        print(f"Model algorithm: {model_name}")

        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model file size: {model_size:.3f}MB")

    except Exception as e:
        print(f"Failed to save model: {output_path}")
        raise e