#!/usr/bin/env python
"""
Pix2Pix/CycleGAN Multi-Key Training Orchestrator

"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

# Hardcoded valid input keys from the dataset
VALID_INPUT_KEYS = ["1_2", "1_5", "1_10", "1_20", "1_50", "1_100"]
TARGET_KEY = "full"


def run_pix2pix_cyclegan_training(
    base_args: List[str],
    input_key: str,
    target_key: str = "full",
    experiment_name: Optional[str] = None,
) -> bool:
    """Run a single pix2pix/cyclegan training session with given input_key."""
    
    # Build training command
    train_args = base_args.copy()
    
    # Update input_key
    if "--input_key" in train_args:
        idx = train_args.index("--input_key")
        train_args[idx + 1] = input_key
    else:
        train_args.extend(["--input_key", input_key])
    
    # Update experiment name if provided
    if experiment_name:
        if "--name" in train_args:
            idx = train_args.index("--name")
            train_args[idx + 1] = experiment_name
        else:
            train_args.extend(["--name", experiment_name])
    
    # Build full command
    pix2pix_path = Path(__file__).parent / "pytorch-CycleGAN-and-pix2pix" / "train.py"
    cmd = ["python", str(pix2pix_path)] + train_args
    
    # Extract model name from args
    model_name = "pix2pix"
    if "--model" in train_args:
        idx = train_args.index("--model")
        model_name = train_args[idx + 1]
    
    print("\n" + "="*80)
    print(f"[Training] Starting {model_name.upper()} training for input_key='{input_key}'")
    if experiment_name:
        print(f"[Training] Experiment name: {experiment_name}")
    print(f"[Training] Target key: {target_key}")
    print("="*80)
    print(f"[Training] Command: {' '.join(cmd)}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"\n✓ Training completed successfully for input_key='{input_key}'")
            return True
        else:
            print(f"\n✗ Training failed for input_key='{input_key}' (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n✗ Error running training for input_key='{input_key}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pix2Pix/CycleGAN Multi-Key Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Orchestrator-specific options
    parser.add_argument(
        "--input-keys",
        nargs="+",
        default=VALID_INPUT_KEYS,
        help=f"Input keys to train on (default: {VALID_INPUT_KEYS})"
    )
    
    # Parse arguments, capturing everything else for pix2pix/cyclegan train.py
    args, pix2pix_args = parser.parse_known_args()
    
    # Validate required arguments
    if "--dataroot" not in pix2pix_args and not any("dataroot" in arg for arg in pix2pix_args):
        parser.error("--dataroot is required")
    
    if "--name" not in pix2pix_args and not any("name" in arg for arg in pix2pix_args):
        parser.error("--name is required")
    
    if "--model" not in pix2pix_args and not any("model" in arg for arg in pix2pix_args):
        parser.error("--model is required (pix2pix or cyclegan)")
    
    # Extract dataroot, name, and model from pix2pix_args
    try:
        idx = pix2pix_args.index("--dataroot")
        dataset_dir = Path(pix2pix_args[idx + 1])
    except ValueError:
        parser.error("Could not parse --dataroot")
    
    try:
        idx = pix2pix_args.index("--name")
        base_name = pix2pix_args[idx + 1]
    except ValueError:
        parser.error("Could not parse --name")
    
    try:
        idx = pix2pix_args.index("--model")
        model_name = pix2pix_args[idx + 1]
    except ValueError:
        parser.error("Could not parse --model")
    
    # Validate requested keys
    input_keys = args.input_keys
    invalid_keys = [k for k in input_keys if k not in VALID_INPUT_KEYS]
    if invalid_keys:
        parser.error(f"Invalid input keys: {invalid_keys}. Valid keys: {VALID_INPUT_KEYS}")
    
    # Train one model per input key
    print(f"\n{'='*80}")
    print(f"[Orchestrator] {model_name.upper()} Multi-Key Training")
    print(f"[Orchestrator] Will train {len(input_keys)} model(s)")
    print(f"[Orchestrator] Input keys: {input_keys}")
    print(f"[Orchestrator] Target key: {TARGET_KEY}")
    print(f"[Orchestrator] Dataset: {dataset_dir}")
    print(f"[Orchestrator] Base name: {base_name}")
    print(f"{'='*80}\n")
    
    results = {}
    for idx, input_key in enumerate(input_keys, 1):
        exp_name = f"{base_name}_{input_key}" if len(input_keys) > 1 else base_name
        
        try:
            success = run_pix2pix_cyclegan_training(
                pix2pix_args,
                input_key=input_key,
                target_key=TARGET_KEY,
                experiment_name=exp_name,
            )
            results[input_key] = success
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt received. Stopping training.")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error for input_key='{input_key}': {e}")
            import traceback
            traceback.print_exc()
            results[input_key] = False
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"[Orchestrator] {model_name.upper()} Training Results")
    print(f"{'='*80}")
    for input_key, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        exp_name = f"{base_name}_{input_key}" if len(input_keys) > 1 else base_name
        print(f"{status}: {input_key} -> {exp_name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    print(f"{'='*80}\n")
    
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
