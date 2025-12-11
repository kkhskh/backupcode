
#!/bin/bash
MODELS="distilgpt2 gpt2 gpt2-medium opt-125m pythia-70m pythia-160m gpt-neo-125m bloom-560m"

for model in $MODELS; do
    echo "=== Testing: $model ==="
    docker rm -f victim 2>/dev/null
    sleep 2
    
    docker run -d --rm --name victim -p 8000:8000 \
        -e MODEL_NAME=$model -e USE_REAL_MODELS=1 \
        -v hf_cache:/root/.cache/huggingface \
        llm-victim
    
    # Wait for health (max 60s)
    for i in {1..30}; do
        curl -s http://localhost:8000/health > /dev/null && break
        sleep 2
    done
    
    python3 traffic_gen.py --mode fingerprint --model-tag $model --n 30
done

docker rm -f victim
python3 analyze_stats.py --files traffic_*.csv --plot results.png
