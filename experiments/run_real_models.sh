#!/bin/bash
MODELS="distilgpt2 gpt2 gpt2-medium opt-125m opt-350m pythia-70m pythia-160m pythia-410m gpt-neo-125m bloom-560m"
REQUESTS=50

for model in $MODELS; do
    echo "=========================================="
    echo "Testing: $model"
    echo "=========================================="
    
    docker stop victim 2>/dev/null
    docker run -d --rm --name victim -p 8000:8000 \
        -e MODEL_NAME=$model -e USE_REAL_MODELS=1 \
        -v hf_cache:/root/.cache/huggingface \
        llm-victim
    
    echo "Waiting for $model to load..."
    sleep 90
    
    # Wait for health
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "Model ready!"
            break
        fi
        sleep 5
    done
    
    python3 traffic_gen.py --mode fingerprint --model-tag $model --n $REQUESTS
done

docker stop victim
echo "Done! Running analysis..."
python3 analyze_stats.py --files traffic_*.csv --plot real_models_fingerprint.png
