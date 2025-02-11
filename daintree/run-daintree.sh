#!/bin/bash

# Written by Claude Sonnet 3.5

# Default values
TARGETS=""
TEST="True"
SKIPFIT="False"

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
    esac
done

docker run --pull=always --rm \
  -v "${PWD}"/output_data:/daintree/output_data \
  -v "${PWD}"/model-map.json:/daintree/model-map.json \
  -v "${PWD}"/sparkles-config:/daintree/sparkles-config \
  us.gcr.io/broad-achilles/daintree:v5 \
  collect-and-fit \
  --input-config model-map.json \
  --sparkles-config /daintree/sparkles-config \
  --out /daintree/output_data \
  --test "$TEST" \
  --skipfit "$SKIPFIT" \
  $([ -n "$TARGETS" ] && echo "--restrict-targets-to $TARGETS")
