#!/bin/bash
MODELS="pythia-70m distilgpt2 gpt2 opt-125m gpt-neo-125m pythia-160m gpt2-medium bloom-560m"
DURATION=30
THRESHOLD=169

cd ~/llm_sidechannel_final

for model in $MODELS; do
    echo "=========================================="
    echo "FLUSH+RELOAD: $model"
    echo "=========================================="
    
    docker rm -f victim 2>/dev/null
    sleep 2
    
    docker run -d --rm --name victim -p 8000:8000 \
        -e MODEL_NAME=$model -e USE_REAL_MODELS=1 \
        -v hf_cache:/root/.cache/huggingface \
        llm-victim
    
    echo "Waiting for $model to load..."
    for i in {1..60}; do
        curl -s http://localhost:8000/health > /dev/null && break
        sleep 2
    done
    sleep 5
    
    echo "Running FLUSH+RELOAD attack..."
    ./attacker/flush_reload 5000 $THRESHOLD 0 > experiments/flush_reload_${model}.csv &
    ATTACKER_PID=$!
    
    for i in {1..50}; do
        curl -s -X POST http://localhost:8000/generate \
            -H "Content-Type: application/json" \
            -d '{"prompt": "The quick brown fox", "max_new_tokens": 20}' > /dev/null
    done
    
    wait $ATTACKER_PID
    echo "Done with $model"
done

docker rm -f victim
echo "All models complete!"
