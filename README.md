# Daintree
## Getting Started
Make sure that you have the following files:
1. Taiga token.
2. Sparkles config file.  
## Usage

To fit a set of predictive models, you can run:  

```
# first create the docker image that will be used to run daintree code locally as well
# as on remote nodes. This step only needs to be done once each time the code changes.
daintree-prep-docker <IMAGE_NAME>

# Submit the job fitting models
daintree-fit-models \
   --image <IMAGE_NAME> \
   --config <MODEL_CONFIG> 
   --out <OUTPUT_DIR>
```
 
Requrired parameters:

* `--image`: The docker image built with this version of daintree which will be used to launch the sparkles job as well as run on the remote worker hosts.
* `--out`: The directory to write the output to.
* `--config`: A JSON file describing how to train the models. See `Model config file` below for a description of the format of this file. You can specify this parameter multiple times for fitting multiple types of models in a single submission.


Optional parameters:
* `--taiga-dir <PATH>`: Path to where Taiga token and cache is stored. (Defaults to "~/.taiga")
* `--sparkles-cache <PATH>`: Path to where sparkles stores cached data. (Defaults to "~/.sparkles-cache")
* `--sparkles-config <PATH>`: Path to sparkles config file (Defaults to ".sparkles)
* `--subset-y <NAME>`: Before submitting jobs, subset the Y matrix to only the following targets (Useful for testing on a subset of data)
* `--skipfit`: This is a boolean flag. Specify if you want to skip the actual model fitting process. (Only useful for testing preprocessing of the input files)

## Model config file
The primary input you need to provide here is the `MODEL_CONFIG` which should be a json file and look similar to the following example:
```
{
    "name": "CellContext",
    "data": {
        "RNASeq": {
            "taiga_filename":"internal-23q4-ac2b.21/OmicsExpressionProteinCodingGenesTPMLogp1",
            "type":"x",
            "required":true,
            "exempt":false,
        },
        "RNAi": {
            "taiga_filename":"demeter2-combined-dc9c.19/gene_means_proc",
            "table_type":"y",
            "relation":"All"
            }
        }
    }
```
This file needs to have at least one feature (specified as `type=x`) matrix and one target (specified as `type=y`) matrix with taiga ids.




-----
# Notes

In other words, I'm proposing a lot of these parameters go away and we offer a top level script so people don't need to run the full docker command.

Parameters that we should be able to hardcode:
```
-w /daintree \ 
--pull=always \
-v "$PWD":/daintree \
/install/depmap-py/bin/python3.9 -u run_fit_models.py collect-and-fit-generate-config \
```

Parameters that we can make into optional parameters and document how people should set:

```
-v ~/.taiga:/root/.taiga \
-v ~/.sparkles-cache:/root/.sparkles-cache \
us.gcr.io/broad-achilles/daintree-sparkles:v1 \
--test True \ # Also, I'm proposing people explictly provide the list of columns they want to use as the test set
--skipfit False \
--sparkles-path /install/sparkles/bin/sparkles \
--sparkles-config /daintree/sparkles-config \
```

The few parameters that I think are reasonable to _require_ users provide because there are no
sensible defaults:
```
--input-files model-map.json \
--save-dir /daintree/output_data/nayeem-test \
```
(I also renamed these `--out` and `--config`)

I was thinking of where to store the results should be outside of daintree so I'd remove this parameter:
```
--upload-to-taiga predictability-76d5
```

When run from Jenkins, we want to upload to GCS. For our prod pipeline, we want it to upload to Taiga. I was imagining that the upload would be an independent step. However,
I can imagine it's convenient to offer this as an option, and we do _already_ have taiga as a dependency, so I guess I'm going to not object too much. However, my instinct is that this will create a little unnecessary coupling. 


In the `model-map` I made the following edits:

I changed `taiga_filename` to `taiga_id` because those names aren't actually files. 

I thought:
```
--model-name CellContext \
```
would really better belong in the `model-map.json` file. We can have it be an optional field so it defaults to "model" if not specified.

Also, it seemed like we were specifing the dataset names (ie: RNASeq, RNAi) twice, so I eliminated the duplication.

I changed `table_type` to `type` and the values to `x` and `y` because "dep" and "feature" are rather specific to depmap, and misleading. (For example, we want to use daintree to fit predictive models for compound sensitivity. But `dep` implies it's something specific to gene dependencies.)

I eliminated "dim_type". What is that used for? `dim_type` in the example was `rnaseq` but if we are processing data specially for specific data sets, I think we should give the value a name more indicative of the processing rather than the dataset name. This will probably need to be added back, but I couldn't even guess what it was for, so I removed it.

We still need documentation for the following fields. I thought these names were pretty good at suggesting what they're for ... but not exactly clear what the implications of changing these is:
```
                    "required":true,
                    "exempt":false,
                "relation":"All"
```

Maybe Lauren would be the best person to document these? (Or is there docs on the Jenkins page?)
