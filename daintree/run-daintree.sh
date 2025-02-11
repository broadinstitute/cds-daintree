#!/bin/bash

# Written by Claude Sonnet 3.5

# Default values
TARGETS=""
TEST="True" # I am keeping it True because I have been only using this bash script for testing purposes and once conseq rules are implemented, I am not sure if this script will be used for production. 
SKIPFIT="False"
INPUT_CONFIG="model-map.json"
UPLOAD_TO_TAIGA=""

# Parse named parameters
for i in "$@"; do
    case $i in
        --targets=*)
        TARGETS="${i#*=}"
        ;;
        --test=*)
        TEST="${i#*=}"
        ;;
        --skipfit=*)
        SKIPFIT="${i#*=}"
        ;;
        --input-config=*)
        INPUT_CONFIG="${i#*=}"
        ;;
        --upload-to-taiga=*)
        UPLOAD_TO_TAIGA="${i#*=}"
        ;;
    esac
done

docker run --pull=always --rm \
  -v "${PWD}"/output_data:/daintree/output_data \
  -v "${PWD}"/"${INPUT_CONFIG}":/daintree/model-map.json \
  -v "${PWD}"/sparkles-config:/daintree/sparkles-config \
  us.gcr.io/broad-achilles/daintree:v5 \
  collect-and-fit \
  --input-config model-map.json \
  --sparkles-config /daintree/sparkles-config \
  --out /daintree/output_data \
  --test "$TEST" \
  --skipfit "$SKIPFIT" \
  $([ -n "$TARGETS" ] && echo "--restrict-targets-to $TARGETS") \
  $([ -n "$UPLOAD_TO_TAIGA" ] && echo "--upload-to-taiga $UPLOAD_TO_TAIGA")
