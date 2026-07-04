#!/bin/bash

# Default directories
DATA_DIR="./data"
RESULTS_DIR="./results/output"
CHECKED_RES_DIR="./results/checked"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_response_data) INPUT_RESPONSE_DATA="$2"; shift ;;
        --output_file_name) OUTPUT_FILE_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Using:"
echo "  INPUT_RESPONSE_DATA: $INPUT_RESPONSE_DATA"
echo "  OUTPUT_FILE_NAME:    $OUTPUT_FILE_NAME"

# Run the Python script
python "./src/check.py" \
    --input_data="$DATA_DIR/data_test.jsonl" \
    --input_response_data="$INPUT_RESPONSE_DATA" \
    --output_dir="$CHECKED_RES_DIR" \
    --output_file_name="$OUTPUT_FILE_NAME" \
    --language="en"
