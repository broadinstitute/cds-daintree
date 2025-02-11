# Daintree
## Getting Started
Make sure that you have the following files:
1. Taiga token.
2. Broad Achilles Secret.
3. Sparkles config file.

## Usage
Once inside the daintree directory, Daintree can be run using the run-daintree.sh script with the following parameters:

```bash
./run-daintree.sh \
  --targets="target1,target2,..." \  # Optional: Comma-separated list of target columns
  --input-config="path/to/file" \   # Optional: Path to model configuration JSON. Will use model-map.json if not specified.
  --test="True|False" \             # Optional: Run in test mode (default: True)
  --skipfit="True|False" \          # Optional: Skip model fitting (default: False)
  --upload-to-taiga="taiga_id"      # Optional: Upload results to Taiga
```
For example, to train model for target genes such as SOX10, PAX8, MDM2, NRAS, KRAS, EBF1, and MYB, run: 
```bash
./run-daintree.sh --targets="SOX10,PAX8,MDM2,NRAS,KRAS,EBF1,MYB"
```
Or, to train model for target drugs, run:
```bash
./run-daintree.sh --targets="PRC-005868644-712-80,PRC-005868644-712-80,PRC-005868644-712-80,PRC-005868644-712-80,PRC-005868644-712-80,PRC-005868644-712-80"
```
Or, to train model using a specific model config file such as model-map-oncref.json and upload results to predictability-76d5, run:
```bash
./run-daintree.sh --input-config="model-map-oncref.json" --upload-to-taiga="predictability-76d5"
```
The bash script can be modified if you would like to use a specific docker image or output to a different directory. By default it uses the `us.gcr.io/broad-achilles/daintree-sparkles:v5` image and outputs to the $(PWD)/output_data directory.

Without the bash script, you can run the following command to train models for example:
```
docker run --rm \
  -v "${PWD}"/model-map.json:/daintree/model-map.json \
  -v "${PWD}"/output_data:/daintree/output_data \
  -v "${PWD}"/sparkles-config:/daintree/sparkles-config \
  us.gcr.io/broad-achilles/daintree:v5 \
  collect-and-fit \
  --input-config model-map.json \
  --sparkles-config /daintree/sparkles-config \
  --out /daintree/output_data \
  --test True \
  --skipfit False \
  --restrict-targets-to "SOX10,PAX8,MDM2,NRAS,KRAS,EBF1,MYB" \
  --upload-to-taiga predictability-76d5
```
#### Parameters
* `--input-config <PATH>`: A JSON file describing how to train the models. See [Model Config File](#model-config-file) below for a description of the format of this file.
* `--sparkles-config <PATH>`: Path to sparkles config file (Defaults to `/daintree/sparkles-config`)
* `--out <PATH>`: The directory to write the output to.
* `--test`: A boolean flag that takes either `True` or `False`. When `True` is selected only a small subset of target variables will be used. (Defaults to `True`)
* `--skipfit`: A boolean flag that takes either `True` or `False`. Specify if you want to skip the actual model fitting process. (Defaults to `True`)
* `--restrict-targets-to`: A comma separated list of target columns to filter. If not provided and `--test` is `True`, uses TEST_LIMIT from `config.py`.
* `--upload-to-taiga`: The taiga id where the output ensemble, feature metadata, and predictions to be uploaded. (Defaults to `None`)

### Model Config File
The primary input you need to provide here is the MODEL_CONFIG which is a JSON file listing the datasets to pull from. It follows the following format:
```
{
  "model_name": [MODEL NAME],
  "screen_name": [SCREEN NAME],
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
**Example**:
```
{
  "model_name": "CellContext",
  "screen_name": "CRISPR",
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
Here `Exempt` is only used for match related. `Required` means that samples that have nans for those features are dropped. This is important because, for example, we want to automatically discard samples that we dont have confounder data for.

This file needs to have at least one feature matrix(specified as `type=feature`) and one target matrix(`type=target_matrix`) with taiga ids.

### File Formats
Both Feature and target datasets/matrices should have samples as the row header/index and feature names as the column headers.

**Example**
```
,SOX10 (6663),NRAS (4893),BRAF (673)
ACH-000001,0,0.5,0.5
ACH-000002,0.6,0.6,0.7
ACH-000003,0.3,0.4,0.4
```

### Output

Once daintree is run, the provided `out` path directory will be populated with the X matrix as a feather file, the downloaded feature matrices as csv files, the feature metadata as `FeatureMetadata{model_name}{screen_name}.csv` file, the predictions as `Predictions{model_name}{screen_name}.csv` file, and the ensemble as `Ensemble{model_name}{screen_name}.csv` file. If a taiga id was provided using the `--upload-to-taiga` flag, the output will also be uploaded to taiga.

## Issues

If you face any issues running daintree, then please reach out to cds-softeng@broadinstitute.org