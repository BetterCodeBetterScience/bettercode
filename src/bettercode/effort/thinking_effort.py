"""
test_thinking_effort.py

Sends a software engineering bug-finding task to Claude Opus with
configurable thinking effort, so you can compare outputs.

Usage:
    python test_thinking_effort.py              # default: high effort
    python test_thinking_effort.py low
    python test_thinking_effort.py medium
    python test_thinking_effort.py high

Requires: ANTHROPIC_API_KEY environment variable set.
"""

import sys
import re
import tempfile
import subprocess
import json
import time
from datetime import datetime
import anthropic
from pathlib import Path

# Load the buggy module source
MODULE_SOURCE = Path(__file__).parent / "median.py"
TEST_SOURCE = Path(__file__).parent / "test_median.py"

PROMPT = """\
I'm building a latency tracking module. I used the standard "Two Heaps" method to find the running median. I implemented "Lazy Deletion" for the remove method to ensure it stays O(log N) instead of O(N).

The Issue:
The median calculation is wrong after I perform removals. It seems biased towards old values. I verified that _clean_top works (it removes deleted items from the root), so I don't understand why the result is still incorrect.

Your task is to identify and fix the bug in the `median.py` module. Please return the complete fixed module code surround by code fences. Do not return just a snippet. 

## median.py

```python
{module_source}
```

"""


def extract_python_code(text, original_module_path, verbose=False):
    """Extract Python code blocks from the response text.
    
    Looks for code blocks marked with ```python or just ``` and returns
    the complete module code. If only a method is provided, it patches
    it into the original module.
    
    Args:
        text: Response text from the model
        original_module_path: Path to the original buggy module
        verbose: If True, print diagnostic information
        
    Returns:
        Complete module code as a string, or None if extraction fails
    """
    # Try to find code blocks with any language identifier or none
    # Matches: ```python\n, ```\n, ```plaintext\n, etc.
    code_blocks = re.findall(r'```[\w]*\s*\n(.*?)```', text, re.DOTALL)
    
    if verbose:
        print(f"  [extract] Found {len(code_blocks)} code blocks (any language)")
        for i, block in enumerate(code_blocks):
            # Show first 50 chars of each block
            preview = block[:50].replace('\n', ' ')
            print(f"    Block {i+1}: {len(block)} chars, starts with: {preview}...")
            print(f"      has class: {'class ' in block}, has def: {'def ' in block}")
    
    if not code_blocks:
        if verbose:
            print(f"  [extract] No code blocks found, trying to extract raw code...")
        
        # Fallback: Check if the entire response looks like Python code
        # Look for class or function definitions at the start
        lines = text.strip().split('\n')
        if lines and (lines[0].startswith('class ') or lines[0].startswith('def ') or 
                     lines[0].startswith('import ') or lines[0].startswith('from ')):
            if verbose:
                print(f"  [extract] Response appears to be raw Python code (starts with: {lines[0][:50]})")
            # Use the entire response as code
            return text.strip()
        
        return None
    
    # Filter to only Python-looking code blocks (have def, class, import, etc.)
    python_blocks = [
        block for block in code_blocks 
        if 'def ' in block or 'class ' in block or 'import ' in block or 'from ' in block
    ]
    
    if verbose:
        print(f"  [extract] {len(python_blocks)} blocks look like Python code")
    
    if not python_blocks:
        # No Python-looking blocks, use all blocks and hope for the best
        python_blocks = code_blocks
        if verbose:
            print(f"  [extract] No Python-specific blocks found, using all code blocks")
    
    # Get the longest block (likely the most complete)
    longest_code = max(python_blocks, key=len)
    
    if verbose:
        print(f"  [extract] Using longest block: {len(longest_code)} chars")
    
    # Check if this looks like a complete module:
    # - Has substantial length (> 500 chars)
    # - Has imports or class definitions or multiple function definitions
    # - Looks complete rather than just a snippet
    has_imports = 'import ' in longest_code or 'from ' in longest_code
    has_class = 'class ' in longest_code
    num_functions = longest_code.count('def ')
    is_substantial = len(longest_code) > 500
    
    looks_complete = (has_imports or has_class or num_functions >= 2) and is_substantial
    
    if verbose:
        print(f"  [extract] Analysis: imports={has_imports}, class={has_class}, "
              f"functions={num_functions}, substantial={is_substantial}")
        print(f"  [extract] Looks complete: {looks_complete}")
    
    # If it looks like a complete module, return it as-is
    if looks_complete:
        if verbose:
            print(f"  [extract] Returning as complete module")
        return longest_code
    
    # Otherwise, it might be just a function fix - attempt to patch it into the original
    if 'def ' in longest_code:
        if verbose:
            print(f"  [extract] Looks like a function snippet, attempting to patch into original")
        
        # Read the original module
        original_code = Path(original_module_path).read_text()
        
        # Try to find the function name being fixed
        func_match = re.search(r'def\s+(\w+)\s*\(', longest_code)
        if func_match:
            func_name = func_match.group(1)
            if verbose:
                print(f"  [extract] Found function: {func_name}")
            
            # Try to find and replace this function in the original
            # Match the entire function including docstring
            pattern = rf'(    def {func_name}\(.*?\n(?:        .*\n)*?)(?=\n    def |\n\ndef |\nclass |\Z)'
            match = re.search(pattern, original_code, re.DOTALL)
            
            if match:
                if verbose:
                    print(f"  [extract] Found {func_name} in original at position {match.start()}-{match.end()}")
                
                # Extract just the method from the fix (handle indentation)
                fixed_function = longest_code.strip()
                
                # Determine indentation level from original
                orig_indent = len(match.group(1)) - len(match.group(1).lstrip())
                current_indent = len(fixed_function) - len(fixed_function.lstrip())
                
                # Adjust indentation if needed
                if orig_indent != current_indent:
                    lines = fixed_function.split('\n')
                    indent_str = ' ' * orig_indent
                    fixed_function = '\n'.join(
                        indent_str + line.lstrip() if line.strip() else line 
                        for line in lines
                    )
                
                # Replace the old function with the new one
                patched_code = original_code[:match.start()] + fixed_function + '\n' + original_code[match.end():]
                if verbose:
                    print(f"  [extract] Patched code: {len(patched_code)} chars")
                return patched_code
            else:
                if verbose:
                    print(f"  [extract] Warning: Could not find {func_name} in original to replace")
    
    # If we can't patch it, just return what we have and hope for the best
    if verbose:
        print(f"  [extract] Returning longest block as-is (could not patch)")
    return longest_code


def test_solution(fixed_code, test_file_path, module_source_path, timestamp, effort, diagnostics_dir, verbose=False):
    """Test the fixed code against the test suite.
    
    Creates a timestamped directory with the fixed code and modified test file
    so you can manually inspect and run the exact test that was performed.
    
    Args:
        fixed_code: The fixed Python code as a string
        test_file_path: Path to the test file
        module_source_path: Path to the original module (to extract module name)
        timestamp: Timestamp string for directory naming
        effort: Effort level string
        diagnostics_dir: Parent diagnostics directory
        verbose: If True, print diagnostic information
        
    Returns:
        tuple: (passed: bool, output: str, test_dir: Path)
    """
    # Get the module name from the source path (e.g., "rover" from "rover.py")
    module_name = Path(module_source_path).stem
    fixed_module_name = f"{module_name}_fixed"
    
    # Create timestamped test directory
    test_dir = diagnostics_dir / f"{timestamp}_{effort}_test"
    test_dir.mkdir(exist_ok=True)
    
    if verbose:
        print(f"  [test] Created test directory: {test_dir.name}/")
        print(f"  [test] Module name: {module_name}")
        print(f"  [test] Fixed module name: {fixed_module_name}")
    
    # Save the fixed code
    fixed_module_path = test_dir / f"{fixed_module_name}.py"
    with open(fixed_module_path, 'w') as f:
        f.write(fixed_code)
    
    if verbose:
        print(f"  [test] Saved fixed code: {fixed_module_path.name}")
    
    # Read original test and replace imports
    original_test = Path(test_file_path).read_text()
    
    # Find what imports exist in the original
    original_imports = [line for line in original_test.split('\n') 
                      if f'import {module_name}' in line or f'from {module_name}' in line]
    
    if verbose and original_imports:
        print(f"  [test] Found {len(original_imports)} import line(s) in test file")
    
    # Replace various import patterns
    modified_test = original_test.replace(
        f'from {module_name} import',
        f'from {fixed_module_name} import'
    ).replace(
        f'import {module_name}',
        f'import {fixed_module_name} as {module_name}'
    )
    
    # Check if any replacements were made
    if verbose:
        if modified_test != original_test:
            print(f"  [test] Imports successfully replaced")
        else:
            print(f"  [test] WARNING: No imports were replaced!")
    
    # Save the modified test
    test_path = test_dir / f"test_{module_name}.py"
    with open(test_path, 'w') as f:
        f.write(modified_test)
    
    if verbose:
        print(f"  [test] Saved modified test: {test_path.name}")
    
    # Create a README to explain the contents
    readme_path = test_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"Test Run: {timestamp} - Effort: {effort}\n")
        f.write("=" * 60 + "\n\n")
        f.write("This directory contains the extracted code and modified test.\n\n")
        f.write("Files:\n")
        f.write(f"  - {fixed_module_path.name}: The fixed code extracted from model response\n")
        f.write(f"  - {test_path.name}: The test file (imports modified to use fixed code)\n\n")
        f.write("To manually run the test:\n")
        f.write(f"  cd {test_dir}\n")
        f.write(f"  pytest {test_path.name} -v\n\n")
        f.write("Or run a specific test:\n")
        f.write(f"  pytest {test_path.name}::TestClassName::test_name -v\n")
    
    if verbose:
        print(f"  [test] Running: pytest {test_path.name} -v --tb=short")
    
    # Run pytest on the test file
    result = subprocess.run(
        ['python', '-m', 'pytest', str(test_path), '-v', '--tb=short'],
        capture_output=True,
        text=True,
        cwd=str(test_dir),  # Run from test directory so imports work
        timeout=30
    )
    
    if verbose:
        print(f"  [test] Pytest exit code: {result.returncode}")
    
    passed = result.returncode == 0
    output = result.stdout + "\n" + result.stderr
    
    return passed, output, test_dir


def save_result(effort, passed, usage, thinking_len, response_len, response_time, extracted_code_len=None):
    """Save the test result to a JSON file.
    
    Args:
        effort: The thinking effort level (low/medium/high)
        passed: Whether all tests passed (bool)
        usage: The API usage stats from the response
        thinking_len: Length of thinking text
        response_len: Length of response text
        response_time: API response time in seconds
        extracted_code_len: Length of extracted code (if any)
    """
    results_file = Path(__file__).parent / "thinking_effort_results.json"
    
    # Load existing results
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    # Add new result
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "effort_level": effort,
        "tests_passed": passed,
        "thinking_length": thinking_len,
        "response_length": response_len,
        "extracted_code_length": extracted_code_len,
        "response_time_seconds": response_time,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
    }
    
    results.append(result_entry)
    
    # Save back to file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Result saved to {results_file.name}")


def main():
    effort = sys.argv[1] if len(sys.argv) > 1 else "high"
    if effort not in ("low", "medium", "high", "max"):
        print(f"Usage: {sys.argv[0]} [low|medium|high|max]")
        sys.exit(1)

    module_source = MODULE_SOURCE.read_text()
    test_source = TEST_SOURCE.read_text()

    prompt = PROMPT.format(module_source=module_source, test_source=test_source)

    client = anthropic.Anthropic()

    print(f"Sending to claude-opus-4-6 with thinking effort: {effort}")
    print(f"Prompt length: ~{len(prompt.split())} words")
    print("-" * 60)

    # Time the API call
    start_time = time.time()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    response_time = time.time() - start_time

    # Print thinking and response separately
    thinking_text = ""
    response_text = ""
    
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
            print(f"\n{'='*60}")
            print(f"THINKING ({len(block.thinking)} chars):")
            print(f"{'='*60}")
            print(block.thinking)
        elif block.type == "text":
            response_text = block.text
            print(f"\n{'='*60}")
            print("RESPONSE:")
            print(f"{'='*60}")
            print(block.text)

    print(f"\n{'='*60}")
    print(f"Usage: {response.usage}")
    print(f"Response time: {response_time:.2f} seconds")

    # Save diagnostic information
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostic_dir = Path(__file__).parent / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    
    # Save thinking and response text
    thinking_file = diagnostic_dir / f"{timestamp}_{effort}_thinking.txt"
    response_file = diagnostic_dir / f"{timestamp}_{effort}_response.txt"
    
    with open(thinking_file, 'w') as f:
        f.write(thinking_text)
    with open(response_file, 'w') as f:
        f.write(response_text)
    
    print(f"\n📝 Diagnostic files saved to {diagnostic_dir.name}/")

    # Test the solution
    print(f"\n{'='*60}")
    print("TESTING SOLUTION:")
    print(f"{'='*60}")
    
    test_passed = False
    extracted_code_len = None
    test_output = ""
    test_dir = None
    
    print("\nExtracting code from response...")
    fixed_code = extract_python_code(response_text, MODULE_SOURCE, verbose=True)
    if fixed_code is None:
        print("❌ Could not extract Python code from the response.")
        print("The response may not contain a complete code solution.")
    else:
        extracted_code_len = len(fixed_code)
        
        # Save extracted code
        extracted_file = diagnostic_dir / f"{timestamp}_{effort}_extracted_code.py"
        with open(extracted_file, 'w') as f:
            f.write(fixed_code)
        
        # Check what we extracted (generic checks)
        has_class = 'class ' in fixed_code
        has_def = 'def ' in fixed_code
        print(f"✓ Extracted {extracted_code_len} characters of Python code")
        print(f"  - Contains class definition: {has_class}")
        print(f"  - Contains function definition: {has_def}")
        print(f"  - Saved to: {extracted_file.name}")
        print("\nRunning tests...")
        
        try:
            test_passed, test_output, test_dir = test_solution(
                fixed_code=fixed_code,
                test_file_path=TEST_SOURCE,
                module_source_path=MODULE_SOURCE,
                timestamp=timestamp,
                effort=effort,
                diagnostics_dir=diagnostic_dir,
                verbose=True
            )
            
            if test_passed:
                print(f"\n{'='*60}")
                print("✅ SUCCESS: All tests passed!")
                print(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print("❌ FAILURE: Some tests failed")
                print(f"{'='*60}")
            
            print("\nTest output:")
            print(test_output)
            print(f"\n📂 Test files saved to: {test_dir.name}/")
            print(f"   You can manually run: cd {test_dir} && pytest -v")
            
        except subprocess.TimeoutExpired:
            print("❌ Tests timed out after 30 seconds")
            test_passed = False
            test_output = "Test execution timed out after 30 seconds"
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            import traceback
            test_output = traceback.format_exc()
            print(test_output)
            test_passed = False
    
    # Save results
    save_result(
        effort=effort,
        passed=test_passed,
        usage=response.usage,
        thinking_len=len(thinking_text),
        response_len=len(response_text),
        response_time=response_time,
        extracted_code_len=extracted_code_len
    )
    
    # Save summary diagnostic file
    summary_file = diagnostic_dir / f"{timestamp}_{effort}_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write(f"THINKING EFFORT DIAGNOSTIC SUMMARY\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Effort Level: {effort}\n")
        f.write(f"Model: claude-opus-4-6\n\n")
        f.write(f"RESULTS:\n")
        f.write(f"  Tests Passed: {test_passed}\n")
        f.write(f"  Response Time: {response_time:.2f}s\n\n")
        f.write(f"CONTENT SIZES:\n")
        f.write(f"  Thinking: {len(thinking_text)} chars\n")
        f.write(f"  Response: {len(response_text)} chars\n")
        f.write(f"  Extracted Code: {extracted_code_len or 0} chars\n\n")
        f.write(f"TOKEN USAGE:\n")
        f.write(f"  Input: {response.usage.input_tokens}\n")
        f.write(f"  Output: {response.usage.output_tokens}\n")
        f.write(f"  Total: {response.usage.input_tokens + response.usage.output_tokens}\n\n")
        f.write(f"FILES SAVED:\n")
        f.write(f"  - {timestamp}_{effort}_thinking.txt\n")
        f.write(f"  - {timestamp}_{effort}_response.txt\n")
        if extracted_code_len:
            f.write(f"  - {timestamp}_{effort}_extracted_code.py\n")
        if test_dir:
            f.write(f"  - {test_dir.name}/  (test directory with runnable code)\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"\nPROMPT EXCERPT:\n")
        f.write(f"{prompt[:500]}...\n\n")
        if test_dir:
            f.write(f"TO MANUALLY RUN THE TEST:\n")
            f.write(f"  cd {test_dir}\n")
            f.write(f"  pytest -v\n\n")
        if not test_passed and test_output:
            f.write(f"TEST FAILURE SUMMARY:\n")
            f.write(f"{test_output[:1000]}...\n")
    
    print(f"📋 Summary saved to: {summary_file.name}")
    print(f"\n💡 To inspect details, check the diagnostics/ folder")



if __name__ == "__main__":
    main()
