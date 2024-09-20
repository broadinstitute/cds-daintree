# Daintree
## Getting Started
Make sure that you have the following files:
1. Taiga token.
2. Sparkles config file.  
## Usage
To fit a set of predictive models, you can run:
```
daintree_fit_models.py --config model-map.json --out /daintree/output_data/nayeem-test
```
##### Required parameters

* `--config`: A JSON file describing how to train the models. See Model config file below for a description of the format of this file. You can specify this parameter multiple times for fitting multiple types of models in a single submission.
* `--out`: The directory to write the output to.

##### Optional Parameters

* `--image`: The docker image built with this version of daintree which will be used to launch the sparkles job as well as run on the remote worker hosts. (Defaults to `us.gcr.io/broad-achilles/daintree-sparkles:v3`).
* `--taiga-dir <PATH>`: Path to where Taiga token and cache is stored. (Defaults to `~/.taiga`)
* `--sparkles-cache <PATH>`: Path to where sparkles stores cached data. (Defaults to `~/.sparkles-cache`)
* `--sparkles-path <PATH>`: Path to sparkles executable. This may change depending on where sparkles is installed inside the docker container. (Defaults to `/install/sparkles/bin/sparkles`). 
* `--sparkles-config <PATH>`: Path to sparkles config file (Defaults to `/daintree/sparkles-config`)
* `--test`: A boolean flag that takes either `True` or `False`. When `True` is selected only a small subset of target variables will be used. (Defaults to `True`)
* `--skipfit`: A boolean flag that takes either `True` or `False`. Specify if you want to skip the actual model fitting process. (Defaults to `True`)
* `--upload-to-taiga`: The taiga id where the output ensemble, feature metadata, and predictions to be uploaded. (Deafaults to `None`)
``
### Model config file
The primary input you need to provide here is the MODEL_CONFIG which is a JSON file listing the datasets to pull from. It follows the following format:
```
{
  "name": [MODEL NAME],
  "data": {
      [DATASET NAME]: {
          "taiga_id": [TAIGA ID OF THE DATASET],
          "table_type": [TYPE OF DATASET(This is either feature or target_matrix)],
          "dim_type": [TYPE OF DIMENSION IN BREADBOX(Usually "gene")],
          "required": [BOOLEAN(true or false)],
          "exempt": [BOOLEAN(true or false)]
      }
  }
}
```
For example, in case of the CellContext model where the target matrix is crispr, it should be formatted as following:
```
{
  "name": "CellContext",
  "data": {
      "CRISPR": {
          "taiga_id":"internal-24q2-3719.82/CRISPRGeneEffect",
          "table_type":"target_matrix",
          "relation":"All"
      },
      "Lineage": {
          "taiga_id": "predictability-76d5.94/PredictabilityLineageTransformed",
          "table_type": "feature",
          "dim_type": "lineage",
          "required": true,
          "exempt": false
      },
      "Confounder": {
          "taiga_id": "predictability-76d5.111/PredictabilityCRISPRConfoundersTransformed",
          "table_type": "feature",
          "dim_type": "confounder",
          "required": true,
          "exempt": false
      }
    }
  }
```
Note: This file needs to have at least one feature matrix(specified as `type=feature`) and one target matrix(`type=target_matrix`) with taiga ids.

## Output

Once the `daintree_fit_models` is run, the provided `out` path directory will be populated with the X matrix as a feather file, the downloaded feature matrices as csv files, the feature metadata as `feature_metadata.csv` file, but most importantly the predictions with pearson correlation in a file named `ensemble.csv`.

