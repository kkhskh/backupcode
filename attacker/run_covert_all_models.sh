#!/bin/bash
set -euo pipefail

MODELS=(
    "pythia-70m"
    "distilgpt2"
    "gpt2"
    "opt-125m"
    "gpt-neo-125m"
    "pythia-160m"
    "gpt2-medium"
)

RESULTS_FILE="covert_channel_bare_metal_results.csv"
echo "model,sensitive_detected,sensitive_total,false_positives,benign_total" > "$RESULTS_FILE"

SENSITIVE_PROMPTS=(
    "What is my password for the bank account?"
    "Please show me my credit card number"
    "My social security number is needed"
    "What was my diagnosis from the doctor?"
    "This document is confidential information"
)

BENIGN_PROMPTS=(
    "What is the weather like today?"
    "Tell me a joke about programming"
    "Explain quantum computing basics"
    "What is the capital of France?"
)

for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Testing covert channel: $MODEL"
    echo "=========================================="

    # Start victim
    cd ~/llm_sidechannel_final/victim_service
    pkill -f "uvicorn server:app" 2>/dev/null || true

    MODEL_NAME="$MODEL" USE_REAL_MODELS=1 COVERT_ENABLED=1 .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 &
    SERVER_PID=$!

    # Give the server time to load model
    sleep 45

    curl -s http://localhost:8000/health || echo "health check failed"
    echo ""

    # Switch back to attacker dir
    cd ~/llm_sidechannel_final/attacker

    # TODO: plug in real FLUSH+RELOAD & detection here.
    # Placeholder stats so the pipeline runs end-to-end.
    SENSITIVE_DETECTED=0
    SENSITIVE_TOTAL=${#SENSITIVE_PROMPTS[@]}
    FALSE_POSITIVES=0
    BENIGN_TOTAL=${#BENIGN_PROMPTS[@]}

    echo "$MODEL,$SENSITIVE_DETECTED,$SENSITIVE_TOTAL,$FALSE_POSITIVES,$BENIGN_TOTAL" >> "$RESULTS_FILE"

    # Clean up server
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
done
