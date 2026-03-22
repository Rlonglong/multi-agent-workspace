#!/bin/bash
# Start Ollama using the external drive to save local space
# The models will be downloaded and stored here
export OLLAMA_MODELS="/Volumes/X9 Pro/long/NYCU/大二上/AI/ollama_models"

# Ensure the directory exists
mkdir -p "$OLLAMA_MODELS"

echo "Starting Ollama with model storage at: $OLLAMA_MODELS"
echo "You can pull models by opening a new terminal and running: ollama pull <model_name>"

# Start the Ollama server
ollama serve
