from .config import FILES
from . import data_processor
from pathlib import Path


def _gather_and_upload(
    save_pref: Path, ipt_dict: dict, model_name: str, screen_name: str, top_n: int
):
    # This could probably be hardcoded and put in a config file. However, I am
    # keeping it this way for now to make it more flexible.
    ensemble_filename = f"Ensemble{model_name}{screen_name}.csv"
    predictions_filename = f"Predictions{model_name}{screen_name}.csv"

    df_ensemble, df_predictions = data_processor.gather_ensemble_tasks(
        save_pref,
        features=str(save_pref / FILES["feature_matrix"]),
        targets=str(save_pref / FILES["target_matrix"]),
        top_n=top_n,
    )
    df_ensemble.to_csv(save_pref / ensemble_filename, index=False)
    df_predictions.to_csv(save_pref / predictions_filename)

    raise Exception("caller needs to handle upload")
    # if upload_to_taiga:
    #     taiga_uploader.upload_results(ipt_dict)
