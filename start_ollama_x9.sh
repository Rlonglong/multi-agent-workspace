#!/bin/bash
# Start Ollama using the external drive to save local space
# The models will be downloaded and stored here
export OLLAMA_MODELS="/Volumes/X9 Pro/long/NYCU/大二上/AI/ollama_models"

# Keep models in RAM for 24 hours so they don't need to reload on every request
export OLLAMA_KEEP_ALIVE=24h

# Only run one model at a time to avoid memory pressure (32B model needs ~22GB)
export OLLAMA_NUM_PARALLEL=1

# Ensure the directory exists
mkdir -p "$OLLAMA_MODELS"

echo "Starting Ollama with model storage at: $OLLAMA_MODELS"
echo "Keep-alive: $OLLAMA_KEEP_ALIVE | Parallel: $OLLAMA_NUM_PARALLEL"
echo "Tip: After starting, pre-warm the model with:"
echo "  curl -s -X POST http://localhost:11434/api/generate -d '{\"model\":\"deepseek-r1:32b\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"24h\"}'"

# Start the Ollama server
ollama serve
