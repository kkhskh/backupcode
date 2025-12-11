
#!/usr/bin/env python3
"""
CPU Memory Fingerprinting Attack - Parallel to Olav's GPU VRAM Approach

Measures:
  - Base memory (model weights loaded)
  - Memory growth per token (KV-cache equivalent on CPU)
  
Varies prompt length: 10, 100, 500, 1000, 2000 tokens
"""

import os
import sys
import time
import json
import subprocess
import requests
import statistics
from typing import List, Dict, Tuple
import csv
from datetime import datetime

# Token lengths to test (like Olav's 2, 1000, 2000, 3000, 4000, 5000)
TOKEN_LENGTHS = [10, 100, 500, 1000, 1500]
TRIALS_PER_LENGTH = 5

# Alphabet that maps to single tokens (from Olav's approach)
SINGLE_TOKEN_CHARS = ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "0."]


def generate_prompt(num_tokens: int) -> str:
    """Generate prompt with exact token count using single-token alphabet."""
    tokens = []
    for i in range(num_tokens):
        tokens.append(SINGLE_TOKEN_CHARS[i % len(SINGLE_TOKEN_CHARS)])
    return " ".join(tokens)


def get_container_memory(container_name: str = "victim") -> float:
    """Get memory usage of container in MB via docker stats."""
    try:
        result = subprocess.run(
            ["docker", "stats", container_name, "--no-stream", "--format", "{{.MemUsage}}"],
            capture_output=True, text=True, timeout=10
        )
        # Parse "1.234GiB / 8GiB" or "500MiB / 8GiB"
        mem_str = result.stdout.strip().split("/")[0].strip()
        if "GiB" in mem_str:
            return float(mem_str.replace("GiB", "")) * 1024
        elif "MiB" in mem_str:
            return float(mem_str.replace("MiB", ""))
        elif "GB" in mem_str:
            return float(mem_str.replace("GB", "")) * 1024
        elif "MB" in mem_str:
            return float(mem_str.replace("MB", ""))
        else:
            return 0.0
    except Exception as e:
        print(f"Error getting memory: {e}")
        return 0.0


def get_system_memory_for_process(process_name: str = "python") -> float:
    """Get memory from /proc for processes matching name."""
    try:
        result = subprocess.run(
            ["bash", "-c", f"ps aux | grep '{process_name}' | grep -v grep | awk '{{sum += $6}} END {{print sum}}'"],
            capture_output=True, text=True, timeout=10
        )
        mem_kb = float(result.stdout.strip() or 0)
        return mem_kb / 1024  # Return MB
    except:
        return 0.0


def measure_memory_during_inference(
    url: str, 
    prompt: str, 
    max_new_tokens: int = 20,
    sample_interval: float = 0.05
) -> Dict:
    """
    Send request and continuously sample memory during inference.
    Returns memory samples and timing.
    """
    memory_samples = []
    
    # Get baseline memory before request
    baseline_mem = get_container_memory()
    
    # Start memory sampling in background
    import threading
    stop_sampling = threading.Event()
    
    def sample_memory():
        while not stop_sampling.is_set():
            mem = get_container_memory()
            memory_samples.append((time.perf_counter(), mem))
            time.sleep(sample_interval)
    
    sampler = threading.Thread(target=sample_memory)
    sampler.start()
    
    # Send request
    start_time = time.perf_counter()
    try:
        response = requests.post(
            url,
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            timeout=120
        )
        success = response.status_code == 200
        if success:
            data = response.json()
            server_time = data.get("elapsed_ms", 0)
        else:
            server_time = 0
    except Exception as e:
        success = False
        server_time = 0
        print(f"Request error: {e}")
    
    end_time = time.perf_counter()
    
    # Stop sampling
    stop_sampling.set()
    sampler.join()
    
    # Get peak memory
    if memory_samples:
        peak_mem = max(s[1] for s in memory_samples)
    else:
        peak_mem = get_container_memory()
    
    return {
        "success": success,
        "baseline_mem_mb": baseline_mem,
        "peak_mem_mb": peak_mem,
        "mem_increase_mb": peak_mem - baseline_mem,
        "elapsed_ms": (end_time - start_time) * 1000,
        "server_ms": server_time,
        "num_samples": len(memory_samples),
        "prompt_tokens": len(prompt.split())
    }


def run_memory_experiment(model: str, url: str = "http://localhost:8000/generate") -> List[Dict]:
    """Run full memory fingerprinting experiment for one model."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"MEMORY FINGERPRINTING: {model}")
    print(f"{'='*60}")
    
    # Get base memory (before any inference)
    base_mem = get_container_memory()
    print(f"Base memory (model loaded): {base_mem:.1f} MB")
    
    for token_len in TOKEN_LENGTHS:
        print(f"\n--- Token length: {token_len} ---")
        prompt = generate_prompt(token_len)
        
        for trial in range(TRIALS_PER_LENGTH):
            result = measure_memory_during_inference(url, prompt, max_new_tokens=50)
            result["model"] = model
            result["target_tokens"] = token_len
            result["trial"] = trial
            result["base_mem_mb"] = base_mem
            results.append(result)
            
            status = "OK" if result["success"] else "FAIL"
            print(f"  Trial {trial+1}: {status} | Peak: {result['peak_mem_mb']:.1f} MB | "
                  f"Δmem: {result['mem_increase_mb']:.1f} MB | Time: {result['elapsed_ms']:.0f}ms")
    
    return results


def analyze_memory_fingerprint(results: List[Dict]) -> Dict:
    """
    Analyze results to extract (base_memory, memory_per_token) fingerprint.
    Like Olav's (A, B) where VRAM(t) = A + B*t
    """
    if not results:
        return {}
    
    model = results[0]["model"]
    base_mem = results[0]["base_mem_mb"]
    
    # Group by token length
    by_length = {}
    for r in results:
        tlen = r["target_tokens"]
        if tlen not in by_length:
            by_length[tlen] = []
        by_length[tlen].append(r["peak_mem_mb"])
    
    # Compute mean peak memory for each length
    lengths = sorted(by_length.keys())
    mean_mems = [statistics.mean(by_length[l]) for l in lengths]
    
    # Linear regression: mem = A + B * tokens
    # A = base memory (intercept)
    # B = memory per token (slope)
    n = len(lengths)
    sum_x = sum(lengths)
    sum_y = sum(mean_mems)
    sum_xy = sum(l * m for l, m in zip(lengths, mean_mems))
    sum_x2 = sum(l * l for l in lengths)
    
    if n * sum_x2 - sum_x * sum_x != 0:
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
    else:
        slope = 0
        intercept = mean_mems[0] if mean_mems else 0
    
    return {
        "model": model,
        "base_mem_mb": base_mem,
        "intercept_mb": intercept,  # A: fitted base memory
        "slope_mb_per_token": slope,  # B: memory growth per token
        "lengths": lengths,
        "mean_mems": mean_mems,
    }


def save_results(results: List[Dict], filename: str):
    """Save raw results to CSV."""
    if not results:
        return
    fieldnames = ["model", "target_tokens", "trial", "success", "base_mem_mb", 
                  "peak_mem_mb", "mem_increase_mb", "elapsed_ms", "server_ms"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to {filename}")


def plot_memory_fingerprint(analyses: List[Dict], output: str):
    """
    Create Olav-style plot:
    - Left: Memory vs Token Length (like Figure 1)
    - Right: Base Memory vs Memory-per-Token scatter (like Figure 2)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(analyses)))
    
    # Left plot: Memory vs Token Length (like Olav's Figure 1)
    ax1 = axes[0]
    for i, a in enumerate(analyses):
        ax1.plot(a["lengths"], a["mean_mems"], 'o-', color=colors[i], 
                label=f"{a['model']}", linewidth=2, markersize=8)
        # Plot regression line
        x_line = np.array([0, max(a["lengths"])])
        y_line = a["intercept_mb"] + a["slope_mb_per_token"] * x_line
        ax1.plot(x_line, y_line, '--', color=colors[i], alpha=0.5)
    
    ax1.set_xlabel("Prompt Length (tokens)", fontsize=12)
    ax1.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax1.set_title("Memory Scaling vs Prompt Length\n(CPU Parallel to GPU VRAM)", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Base Memory vs Memory-per-Token (like Olav's Figure 2)
    ax2 = axes[1]
    for i, a in enumerate(analyses):
        ax2.scatter(a["intercept_mb"], a["slope_mb_per_token"] * 1000,  # Convert to MB/1000 tokens
                   s=200, c=[colors[i]], label=a["model"], edgecolors='black', linewidth=1)
        ax2.annotate(a["model"], (a["intercept_mb"], a["slope_mb_per_token"] * 1000),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel("Base Memory Allocated (MB)", fontsize=12)
    ax2.set_ylabel("Marginal Memory per 1000 Tokens (MB)", fontsize=12)
    ax2.set_title("CPU Memory Fingerprint\n(Intercept vs Slope)", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CPU Memory Fingerprinting Attack")
    parser.add_argument("--models", "-m", nargs="+", 
                       default=["distilgpt2", "gpt2", "gpt2-medium", "pythia-70m", "bloom-560m"])
    parser.add_argument("--output", "-o", default="memory_fingerprint")
    parser.add_argument("--plot", "-p", default="memory_fingerprint.png")
    parser.add_argument("--url", default="http://localhost:8000/generate")
    args = parser.parse_args()
    
    all_results = []
    all_analyses = []
    
    for model in args.models:
        print(f"\n{'#'*60}")
        print(f"Starting model: {model}")
        print(f"{'#'*60}")
        
        # Stop any existing container
        subprocess.run(["docker", "rm", "-f", "victim"], capture_output=True)
        time.sleep(2)
        
        # Start container
        cmd = [
            "docker", "run", "-d", "--rm", "--name", "victim",
            "-p", "8000:8000",
            "-e", f"MODEL_NAME={model}",
            "-e", "USE_REAL_MODELS=1",
            "-v", "hf_cache:/root/.cache/huggingface",
            "llm-victim"
        ]
        subprocess.run(cmd)
        
        # Wait for model to load
        print("Waiting for model to load...")
        for i in range(60):
            try:
                r = requests.get("http://localhost:8000/health", timeout=5)
                if r.status_code == 200:
                    print("Model ready!")
                    break
            except:
                pass
            time.sleep(2)
        else:
            print("Timeout waiting for model")
            continue
        
        # Extra wait for memory to stabilize
        time.sleep(5)
        
        # Run experiment
        results = run_memory_experiment(model, args.url)
        all_results.extend(results)
        
        # Analyze
        analysis = analyze_memory_fingerprint(results)
        all_analyses.append(analysis)
        
        print(f"\n{model} Fingerprint:")
        print(f"  Base Memory (intercept): {analysis['intercept_mb']:.1f} MB")
        print(f"  Memory/Token (slope): {analysis['slope_mb_per_token']*1000:.3f} MB/1000 tokens")
    
    # Stop container
    subprocess.run(["docker", "rm", "-f", "victim"], capture_output=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, f"{args.output}_{timestamp}.csv")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("MEMORY FINGERPRINT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Base Mem (MB)':<15} {'Slope (MB/1K tok)':<20}")
    print("-"*70)
    for a in all_analyses:
        print(f"{a['model']:<20} {a['intercept_mb']:<15.1f} {a['slope_mb_per_token']*1000:<20.4f}")
    
    # Plot
    if all_analyses:
        plot_memory_fingerprint(all_analyses, args.plot)


if __name__ == "__main__":
    main()
