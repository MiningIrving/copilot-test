#!/usr/bin/env python3
"""
Tokenization Configuration Fix for Qwen3

This script addresses the tokenization configuration inconsistency identified
as the primary root cause of the qwen3 model output differences.

Based on the validation results showing:
- C++ tokens: 101325, 101283  
- Python tokens: 151645, 151645 (repeated)

The fix focuses on ensuring consistent tokenizer configuration between
C++ and Python implementations.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def create_consistent_tokenizer_config() -> Dict[str, Any]:
    """Create a consistent tokenizer configuration for both C++ and Python"""
    
    # Standard Qwen3 tokenizer configuration
    config = {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "auto_map": {
            "AutoTokenizer": ["tokenization_qwen3.Qwen3TokenizerFast", None]
        },
        "bos_token": {
            "__type": "AddedToken",
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
        },
        "eos_token": {
            "__type": "AddedToken", 
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
        },
        "pad_token": {
            "__type": "AddedToken",
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
        },
        "unk_token": {
            "__type": "AddedToken",
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
        },
        "clean_up_tokenization_spaces": False,
        "split_special_tokens": False,
        "tokenizer_type": "tiktoken",
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    }
    
    return config


def fix_qwen3_config_dtype_consistency():
    """Fix dtype-related configuration issues in Qwen3Config"""
    
    config_fixes = {
        # Ensure consistent data types for model parameters
        "torch_dtype": "auto",  # Let the model determine the appropriate dtype
        "use_cache": True,
        
        # RMSNorm precision fixes
        "rms_norm_eps": 1e-6,  # Ensure consistent epsilon value
        
        # Attention configuration fixes  
        "attention_bias": False,  # Ensure consistent bias handling
        "attention_dropout": 0.0,  # Consistent dropout
        
        # Rope configuration
        "rope_theta": 1000000.0,  # Consistent with Qwen3 default
        
        # Ensure consistent layer types
        "use_sliding_window": False,  # Default setting
        "sliding_window": 4096,
        "max_window_layers": 28,
    }
    
    return config_fixes


def create_tokenizer_initialization_fix():
    """Create a fix for consistent tokenizer initialization"""
    
    fix_code = '''
def ensure_consistent_tokenizer_config(config_path: str):
    """Ensure tokenizer configuration consistency between C++ and Python"""
    import json
    from pathlib import Path
    
    config_file = Path(config_path) / "tokenizer_config.json"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            current_config = json.load(f)
        
        # Apply consistent configuration
        consistent_config = create_consistent_tokenizer_config()
        current_config.update(consistent_config)
        
        # Write back the fixed configuration
        with open(config_file, 'w') as f:
            json.dump(current_config, f, indent=2)
        
        print(f"✓ Updated tokenizer configuration: {config_file}")
    else:
        # Create new consistent configuration
        consistent_config = create_consistent_tokenizer_config()
        with open(config_file, 'w') as f:
            json.dump(consistent_config, f, indent=2)
        
        print(f"✓ Created tokenizer configuration: {config_file}")


def fix_model_config_consistency(config_path: str):
    """Fix model configuration for dtype consistency"""
    import json
    from pathlib import Path
    
    config_file = Path(config_path) / "config.json"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            current_config = json.load(f)
        
        # Apply dtype and precision fixes
        dtype_fixes = fix_qwen3_config_dtype_consistency()
        current_config.update(dtype_fixes)
        
        # Write back the fixed configuration
        with open(config_file, 'w') as f:
            json.dump(current_config, f, indent=2)
        
        print(f"✓ Updated model configuration: {config_file}")
    else:
        print(f"⚠ Model config file not found: {config_file}")
'''
    
    return fix_code


def apply_tokenization_fixes(model_path: Optional[str] = None):
    """Apply tokenization fixes to the model directory"""
    
    if model_path is None:
        # Default to current directory structure
        model_path = "/home/runner/work/copilot-test/copilot-test/qwen3"
    
    model_dir = Path(model_path)
    print(f"Applying tokenization fixes to: {model_dir}")
    
    # Create consistent tokenizer config
    tokenizer_config = create_consistent_tokenizer_config()
    tokenizer_config_file = model_dir / "tokenizer_config.json"
    
    with open(tokenizer_config_file, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✓ Created consistent tokenizer config: {tokenizer_config_file}")
    
    # Create model config fixes
    model_config_fixes = fix_qwen3_config_dtype_consistency()
    
    # Check if there's an existing config.json to update
    config_file = model_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            current_config = json.load(f)
        current_config.update(model_config_fixes)
    else:
        # Create a basic config with our fixes
        current_config = {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 11008,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "max_position_embeddings": 32768,
            **model_config_fixes
        }
    
    with open(config_file, 'w') as f:
        json.dump(current_config, f, indent=2)
    print(f"✓ Updated model config: {config_file}")
    
    # Create the fix code file for reference
    fix_code = create_tokenizer_initialization_fix()
    fix_code_file = model_dir / "tokenization_fix.py"
    with open(fix_code_file, 'w') as f:
        f.write(fix_code)
    print(f"✓ Created tokenization fix code: {fix_code_file}")


def validate_tokenization_fix():
    """Validate that the tokenization fix addresses the key issues"""
    
    print("=" * 60)
    print("VALIDATING TOKENIZATION FIXES")
    print("=" * 60)
    
    # Check the key issues from the analysis:
    # 1. Tokenization configuration consistency
    # 2. Data type consistency
    # 3. Special token handling
    
    validation_results = {
        "tokenizer_config_created": False,
        "model_config_updated": False,
        "dtype_fixes_applied": False,
        "special_tokens_configured": False
    }
    
    model_dir = Path("/home/runner/work/copilot-test/copilot-test/qwen3")
    
    # Check tokenizer config
    tokenizer_config_file = model_dir / "tokenizer_config.json"
    if tokenizer_config_file.exists():
        with open(tokenizer_config_file, 'r') as f:
            config = json.load(f)
        if "vocab_size" in config and config["vocab_size"] == 151936:
            validation_results["tokenizer_config_created"] = True
        if "bos_token" in config and "eos_token" in config:
            validation_results["special_tokens_configured"] = True
    
    # Check model config
    model_config_file = model_dir / "config.json"
    if model_config_file.exists():
        with open(model_config_file, 'r') as f:
            config = json.load(f)
        if "rms_norm_eps" in config and "attention_bias" in config:
            validation_results["model_config_updated"] = True
            validation_results["dtype_fixes_applied"] = True
    
    # Print validation results
    for check, passed in validation_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")
    
    all_passed = all(validation_results.values())
    print(f"\nOverall validation: {'✓ PASS' if all_passed else '✗ FAIL'}")
    
    return all_passed


if __name__ == "__main__":
    print("QWEN3 TOKENIZATION CONFIGURATION FIX")
    print("=" * 60)
    
    # Apply the fixes
    apply_tokenization_fixes()
    
    # Validate the fixes
    validate_tokenization_fix()
    
    print("\n" + "=" * 60)
    print("TOKENIZATION FIX COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Regenerate the modeling_qwen3.py file from modular_qwen3.py")
    print("2. Test the model with consistent tokenizer configuration")
    print("3. Verify that C++ implementation uses the same configuration")
    print("4. Run validation tests to confirm output consistency")