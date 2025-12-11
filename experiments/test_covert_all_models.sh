
#!/bin/bash
MODELS="pythia-70m distilgpt2 gpt2 opt-125m gpt-neo-125m pythia-160m gpt2-medium bloom-560m"

echo "=== COVERT CHANNEL: ALL MODELS ==="

for model in $MODELS; do
    echo ""
    echo "--- $model ---"
    docker rm -f victim 2>/dev/null
    
    docker run -d --rm --name victim -p 8000:8000 \
        -e MODEL_NAME=$model -e USE_REAL_MODELS=1 \
        -e COVERT_ENABLED=1 \
        -v hf_cache:/root/.cache/huggingface \
        llm-victim
    
    # Wait for model
    for i in {1..60}; do
        curl -s -X POST http://localhost:8000/generate \
            -H "Content-Type: application/json" \
            -d '{"prompt": "test", "max_new_tokens": 1}' > /dev/null 2>&1 && break
        sleep 2
    done
    
    # Test sensitive
    RESULT=$(curl -s -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "What is my password?", "max_new_tokens": 1}')
    TRIGGERED=$(echo $RESULT | grep -o '"covert_triggered":true' | wc -l)
    
    # Test benign
    RESULT2=$(curl -s -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "What is the weather?", "max_new_tokens": 1}')
    FALSE_POS=$(echo $RESULT2 | grep -o '"covert_triggered":true' | wc -l)
    
    echo "$model: sensitive=$TRIGGERED/1, false_pos=$FALSE_POS/1"
done

docker rm -f victim
echo ""
echo "=== COMPLETE ==="
