# Daintree
## Getting Started
Make sure that you have the following files:
1. Taiga token.
2. Sparkles config file.  
## Usage
Now to run the fit predictive model, you can run: 
`docker run -w /daintree \
            --pull=always \
            -v "$PWD":/daintree \
            -v ~/.taiga:/root/.taiga \
            -v ~/.sparkles-cache:/root/.sparkles-cache \
            us.gcr.io/broad-achilles/daintree-sparkles:v1 \
            /install/depmap-py/bin/python3.9 -u run_fit_models.py collect-and-fit-generate-config \
            --input-files model-map.json \
            --sparkles-path /install/sparkles/bin/sparkles \
            --sparkles-config /daintree/sparkles-config \
            --save-dir /daintree/output_data/nayeem-test \
            --test True \
            --model-name CellContext \
            --skipfit False \
            --upload-to-taiga predictability-76d5`

The primary input you need to provide here is the `model-map.json` which looks like the following:
`{
    "RNASeq": {"name":"RNASeq",
                "taiga_filename":"internal-23q4-ac2b.21/OmicsExpressionProteinCodingGenesTPMLogp1",
                "table_type":"feature",
                "required":true,
                "exempt":false,
                "dim_type": "rnaseq"
            },
    "RNAi": {"name":"RNAi",
            "taiga_filename":"demeter2-combined-dc9c.19/gene_means_proc",
            "table_type":"dep",
            "relation":"All"}
    }`
This file needs to have at least one feature matrix and one target matrix with taiga ids. The other important flags are following:

* `--sparkles-config`: Your local sparkles config path.
* `--save-dir`: The location of the output to save.
* `--test`: If you want to run at a small scale on a test basis, then choose True. You can configure how many genes you want.
* `--model-name`: The name of the model. The default name is "Model".
* `--skipfit`: This is a boolean flag. If you don't want to run the fit function then choose True. 
* `--upload-to-taiga`: The taiga id where the output matrix and feature metadata to be uploaded. 

