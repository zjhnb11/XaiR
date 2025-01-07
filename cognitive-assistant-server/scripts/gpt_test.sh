#!/bin/bash

#conda activate XaiR
# Extract the API key from constant.py
OPENAI_API_KEY=$(grep -oP 'API_KEY\s*=\s*"\K[^"]+' constants.py)

# Export the API key
export OPENAI_API_KEY

#echo $OPENAI_API_KEY
# Run the gpt_api_test.py script
python3 gpt_api_test.py