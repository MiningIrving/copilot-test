
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
