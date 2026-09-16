import pandas as pd
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

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        print(f"Failed to save model results for {model_name} because there are only {n_clusters} clusters.")
        return results

    model_silhouette = silhouette_score(x, labels)
    model_davies = davies_bouldin_score(x, labels)
    model_calinski = calinski_harabasz_score(x, labels)

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