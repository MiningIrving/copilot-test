#!/usr/bin/env python3
"""
Qwen3 Tokenization Debugging Tool

This script specifically focuses on debugging the tokenization issues
identified in the validation results. Since tokenization is failing,
it's the primary root cause of all downstream issues.

The script creates detailed tests to identify where the tokenization
pipeline differs between C++ and Python implementations.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers library available")
except ImportError:
    print("⚠ Transformers library not available")
    TRANSFORMERS_AVAILABLE = False


def analyze_tokenization_discrepancy():
    """Analyze the specific tokenization discrepancy from validation results"""
    
    print("=" * 80)
    print("TOKENIZATION DISCREPANCY ANALYSIS")
    print("=" * 80)
    
    # From the validation results, we know:
    # - C++ tokens: 101325, 101283
    # - Python tokens: 151645, 151645 (repeated)
    
    cpp_tokens = [101325, 101283]
    python_tokens = [151645, 151645]
    
    print("Observed token differences:")
    print(f"  C++ tokens:    {cpp_tokens}")
    print(f"  Python tokens: {python_tokens}")
    print()
    
    # Analysis of the differences
    print("Analysis:")
    print(f"  1. C++ produces different tokens each step: {len(set(cpp_tokens))} unique")
    print(f"  2. Python produces same token repeatedly: {len(set(python_tokens))} unique")
    print(f"  3. Token value difference: C++={cpp_tokens[0]}, Python={python_tokens[0]}")
    print(f"  4. Difference magnitude: {abs(cpp_tokens[0] - python_tokens[0])}")
    print()
    
    # Check if tokens are in valid range for Qwen3
    qwen3_vocab_size = 151936
    print("Token validity check:")
    for i, (cpp_token, py_token) in enumerate(zip(cpp_tokens, python_tokens)):
        cpp_valid = 0 <= cpp_token < qwen3_vocab_size
        py_valid = 0 <= py_token < qwen3_vocab_size
        print(f"  Step {i}: C++ token {cpp_token} valid: {cpp_valid}, Python token {py_token} valid: {py_valid}")
    
    # Hypothesis about the root cause
    print()
    print("ROOT CAUSE HYPOTHESES:")
    print("1. Different tokenizer configurations/vocabularies being used")
    print("2. Different input preprocessing (text encoding, normalization)")
    print("3. Different special token handling (BOS, EOS, PAD)")
    print("4. Model weight loading issues affecting embedding lookup")
    print("5. Different model architecture configurations")
    print()
    
    return {
        'cpp_tokens': cpp_tokens,
        'python_tokens': python_tokens,
        'vocab_size': qwen3_vocab_size,
        'hypotheses': [
            'tokenizer_config_mismatch',
            'input_preprocessing_difference', 
            'special_token_handling',
            'weight_loading_issues',
            'architecture_config_mismatch'
        ]
    }


def test_qwen3_tokenizer_configurations():
    """Test different Qwen3 tokenizer configurations to identify issues"""
    
    print("=" * 80)
    print("TESTING QWEN3 TOKENIZER CONFIGURATIONS")
    print("=" * 80)
    
    if not TRANSFORMERS_AVAILABLE:
        print("⚠ Cannot test tokenizers - transformers library not available")
        return {}
    
    # Test different model paths and configurations
    test_cases = [
        {"name": "qwen3_official", "model_id": "Qwen/Qwen3-8B"},
        {"name": "qwen3_chat", "model_id": "Qwen/Qwen3-8B-Chat"},
        {"name": "qwen3_1.5b", "model_id": "Qwen/Qwen3-1.5B"},
    ]
    
    test_strings = [
        "Hello, how are you?",
        "你好，你怎么样？",
        "This is a test.",
        "",  # Empty string
        " ",  # Single space
        "\n",  # Newline
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"Testing {case['name']} ({case['model_id']})...")
        
        try:
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(case['model_id'], trust_remote_code=True)
            
            case_results = {
                'vocab_size': tokenizer.vocab_size,
                'special_tokens': {
                    'bos_token': tokenizer.bos_token,
                    'eos_token': tokenizer.eos_token,
                    'pad_token': tokenizer.pad_token,
                    'unk_token': tokenizer.unk_token,
                },
                'special_token_ids': {
                    'bos_token_id': tokenizer.bos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'pad_token_id': tokenizer.pad_token_id,
                    'unk_token_id': getattr(tokenizer, 'unk_token_id', None),
                },
                'tokenization_tests': {}
            }
            
            # Test tokenization of different strings
            for test_string in test_strings:
                try:
                    tokens = tokenizer.encode(test_string, add_special_tokens=False)
                    tokens_with_special = tokenizer.encode(test_string, add_special_tokens=True)
                    
                    case_results['tokenization_tests'][repr(test_string)] = {
                        'tokens_no_special': tokens,
                        'tokens_with_special': tokens_with_special,
                        'decoded': tokenizer.decode(tokens, skip_special_tokens=True),
                        'decoded_with_special': tokenizer.decode(tokens_with_special, skip_special_tokens=False)
                    }
                    
                except Exception as e:
                    case_results['tokenization_tests'][repr(test_string)] = {
                        'error': str(e)
                    }
            
            results[case['name']] = case_results
            print(f"  ✓ Successfully tested {case['name']}")
            print(f"    Vocab size: {case_results['vocab_size']}")
            print(f"    BOS/EOS/PAD: {case_results['special_token_ids']['bos_token_id']}/{case_results['special_token_ids']['eos_token_id']}/{case_results['special_token_ids']['pad_token_id']}")
            
        except Exception as e:
            results[case['name']] = {'error': str(e)}
            print(f"  ✗ Failed to test {case['name']}: {e}")
    
    return results


def create_tokenization_test_vectors():
    """Create test vectors specifically for debugging tokenization issues"""
    
    print("=" * 80)
    print("CREATING TOKENIZATION TEST VECTORS")
    print("=" * 80)
    
    # Create test cases that might expose tokenization differences
    test_vectors = {}
    
    # Basic test cases
    basic_tests = [
        "Hello",
        "Hello world",
        "Hello, world!",
        "你好",
        "你好世界",
        "Hello 你好",
        "",
        " ",
        "\t",
        "\n",
        "123",
        "!@#$%^&*()",
    ]
    
    # Edge cases that might cause issues
    edge_cases = [
        # Unicode edge cases
        "\u0000",  # Null character
        "\uffff",  # Max BMP character
        "𝕳𝖊𝖑𝖑𝖔",  # Mathematical characters
        
        # Long strings
        "A" * 1000,
        "你好" * 500,
        
        # Mixed scripts
        "Hello世界123!@#",
        
        # Special sequences that might be interpreted differently
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[UNK]",
    ]
    
    # Combine all test cases
    all_test_cases = basic_tests + edge_cases
    
    print(f"Creating {len(all_test_cases)} tokenization test vectors...")
    
    for i, test_string in enumerate(all_test_cases):
        test_vectors[f'test_{i:03d}'] = {
            'input_string': test_string,
            'description': f'Test case {i}: {repr(test_string)[:50]}...' if len(repr(test_string)) > 50 else f'Test case {i}: {repr(test_string)}',
            'expected_issues': []
        }
        
        # Add expected issues for specific cases
        if not test_string:
            test_vectors[f'test_{i:03d}']['expected_issues'].append('empty_string')
        if test_string.isspace():
            test_vectors[f'test_{i:03d}']['expected_issues'].append('whitespace_only')
        if len(test_string) > 100:
            test_vectors[f'test_{i:03d}']['expected_issues'].append('long_string')
        if any(ord(c) > 127 for c in test_string):
            test_vectors[f'test_{i:03d}']['expected_issues'].append('non_ascii')
        if any(c in '<>[]|' for c in test_string):
            test_vectors[f'test_{i:03d}']['expected_issues'].append('special_tokens')
    
    return test_vectors


def create_embedding_lookup_tests():
    """Create tests for embedding lookup issues"""
    
    print("=" * 80)
    print("CREATING EMBEDDING LOOKUP TESTS")
    print("=" * 80)
    
    # Based on the observed token differences, create tests
    # C++ tokens: 101325, 101283
    # Python tokens: 151645, 151645
    
    vocab_size = 151936
    
    test_cases = {}
    
    # Test the specific problematic tokens
    problematic_tokens = [101325, 101283, 151645]
    
    for token_id in problematic_tokens:
        test_cases[f'token_{token_id}'] = {
            'token_id': token_id,
            'valid': 0 <= token_id < vocab_size,
            'description': f'Test embedding lookup for token {token_id}'
        }
    
    # Test boundary cases
    boundary_tokens = [0, 1, vocab_size - 2, vocab_size - 1]
    
    for token_id in boundary_tokens:
        test_cases[f'boundary_{token_id}'] = {
            'token_id': token_id,
            'valid': 0 <= token_id < vocab_size,
            'description': f'Test boundary token {token_id}'
        }
    
    # Test sequences
    sequences = [
        [101325, 101283],  # C++ sequence
        [151645, 151645],  # Python sequence
        [0, 1, 2],         # Start of vocab
        [vocab_size-3, vocab_size-2, vocab_size-1],  # End of vocab
    ]
    
    for i, seq in enumerate(sequences):
        test_cases[f'sequence_{i}'] = {
            'token_ids': seq,
            'valid': all(0 <= tid < vocab_size for tid in seq),
            'description': f'Test sequence {seq}'
        }
    
    return test_cases


def generate_tokenization_debugging_plan():
    """Generate a specific plan for debugging tokenization issues"""
    
    plan = []
    plan.append("=" * 80)
    plan.append("TOKENIZATION DEBUGGING PLAN")
    plan.append("=" * 80)
    plan.append("")
    
    plan.append("PHASE 1: CONFIGURATION VERIFICATION")
    plan.append("1. Compare tokenizer configurations between C++ and Python")
    plan.append("   - Verify same vocabulary file is being used")
    plan.append("   - Check special token definitions")
    plan.append("   - Verify tokenizer parameters (normalization, etc.)")
    plan.append("")
    
    plan.append("2. Test with identical input strings")
    plan.append("   - Use the test vectors created above")
    plan.append("   - Compare token outputs step by step")
    plan.append("   - Look for systematic differences")
    plan.append("")
    
    plan.append("PHASE 2: INPUT PROCESSING VERIFICATION")
    plan.append("1. Check input preprocessing pipeline")
    plan.append("   - Text encoding (UTF-8, etc.)")
    plan.append("   - Normalization steps")
    plan.append("   - Special character handling")
    plan.append("")
    
    plan.append("2. Verify tokenization algorithm")
    plan.append("   - BPE/SentencePiece implementation")
    plan.append("   - Merge rules application")
    plan.append("   - Unknown token handling")
    plan.append("")
    
    plan.append("PHASE 3: MODEL INTEGRATION VERIFICATION")
    plan.append("1. Check embedding lookup")
    plan.append("   - Verify embedding weights are loaded correctly")
    plan.append("   - Test specific token embeddings")
    plan.append("   - Compare embedding outputs for same tokens")
    plan.append("")
    
    plan.append("2. Verify generation pipeline")
    plan.append("   - Check if issue is in forward pass vs. generation")
    plan.append("   - Test with known good tokens")
    plan.append("   - Verify sampling/selection logic")
    plan.append("")
    
    plan.append("IMMEDIATE ACTIONS:")
    plan.append("1. Create minimal reproduction case with specific failing tokens")
    plan.append("2. Compare tokenizer outputs for identical input")
    plan.append("3. Test embedding lookups for tokens 101325, 101283, 151645")
    plan.append("4. Verify model configuration consistency")
    plan.append("5. Check if C++ is using a different model/weights than Python")
    plan.append("")
    
    return "\n".join(plan)


def main():
    """Main entry point"""
    
    print("QWEN3 TOKENIZATION DEBUGGING TOOL")
    print("Focusing on the primary root cause: tokenization failure")
    print()
    
    # Analyze the specific discrepancy
    discrepancy_analysis = analyze_tokenization_discrepancy()
    
    # Test tokenizer configurations (if available)
    tokenizer_tests = test_qwen3_tokenizer_configurations()
    
    # Create test vectors
    test_vectors = create_tokenization_test_vectors()
    
    # Create embedding tests
    embedding_tests = create_embedding_lookup_tests()
    
    # Generate debugging plan
    debugging_plan = generate_tokenization_debugging_plan()
    
    # Save all results
    output_dir = Path("tokenization_debugging")
    output_dir.mkdir(exist_ok=True)
    
    # Save analysis
    with open(output_dir / "discrepancy_analysis.json", 'w') as f:
        json.dump(discrepancy_analysis, f, indent=2)
    
    # Save tokenizer tests
    with open(output_dir / "tokenizer_tests.json", 'w') as f:
        json.dump(tokenizer_tests, f, indent=2)
    
    # Save test vectors
    with open(output_dir / "test_vectors.json", 'w') as f:
        json.dump(test_vectors, f, indent=2)
    
    # Save embedding tests
    with open(output_dir / "embedding_tests.json", 'w') as f:
        json.dump(embedding_tests, f, indent=2)
    
    # Save debugging plan
    with open(output_dir / "debugging_plan.txt", 'w') as f:
        f.write(debugging_plan)
    
    # Display plan
    print(debugging_plan)
    
    print(f"All debugging materials saved to: {output_dir}/")
    print()
    print("KEY FINDINGS:")
    print(f"- C++ tokens: {discrepancy_analysis['cpp_tokens']}")
    print(f"- Python tokens: {discrepancy_analysis['python_tokens']}")
    print("- Python produces the same token repeatedly (likely generation issue)")
    print("- C++ produces valid but different tokens (likely tokenization/config issue)")
    print()
    print("RECOMMENDED NEXT STEP:")
    print("Focus on verifying tokenizer configuration consistency between C++ and Python")


if __name__ == "__main__":
    main()