#!/usr/bin/env python3
"""
Check version consistency between pyproject.toml and .conda/meta.yaml for pylibsparseir dependency.
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple


def extract_version_from_pyproject_toml(file_path: Path) -> Optional[str]:
    """Extract pylibsparseir version from pyproject.toml dependencies."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for pylibsparseir in dependencies array
        pattern = r'["\']pylibsparseir([^"\']+)["\']'
        match = re.search(pattern, content)
        
        if match:
            return match.group(1)
        else:
            print(f"Warning: pylibsparseir not found in {file_path}")
            return None
            
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_version_from_meta_yaml(file_path: Path) -> Optional[str]:
    """Extract pylibsparseir version from .conda/meta.yaml requirements."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for spm-lab::pylibsparseir in run requirements
        pattern = r'spm-lab::pylibsparseir\s+([^\s]+)'
        match = re.search(pattern, content)
        
        if match:
            return match.group(1)
        else:
            print(f"Warning: spm-lab::pylibsparseir not found in {file_path}")
            return None
            
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def normalize_version_spec(version_spec: str) -> str:
    """Normalize version specification for comparison."""
    # Remove any whitespace
    version_spec = version_spec.strip()
    
    # For now, just return as-is since both should use the same format
    # In the future, we could add more sophisticated normalization
    return version_spec


def check_version_consistency() -> bool:
    """Check if pylibsparseir versions are consistent between pyproject.toml and .conda/meta.yaml."""
    project_root = Path(__file__).parent
    pyproject_path = project_root / "pyproject.toml"
    meta_yaml_path = project_root / ".conda" / "meta.yaml"
    
    print("Checking version consistency for pylibsparseir dependency...")
    print(f"pyproject.toml: {pyproject_path}")
    print(f".conda/meta.yaml: {meta_yaml_path}")
    print()
    
    # Extract versions
    pyproject_version = extract_version_from_pyproject_toml(pyproject_path)
    meta_yaml_version = extract_version_from_meta_yaml(meta_yaml_path)
    
    if pyproject_version is None or meta_yaml_version is None:
        print("❌ Failed to extract version information from one or both files")
        return False
    
    # Normalize versions for comparison
    pyproject_normalized = normalize_version_spec(pyproject_version)
    meta_yaml_normalized = normalize_version_spec(meta_yaml_version)
    
    print(f"pyproject.toml version: {pyproject_version}")
    print(f".conda/meta.yaml version: {meta_yaml_version}")
    print()
    
    # Show the actual lines where the versions are found
    print("Found in files:")
    if pyproject_version:
        print(f"  pyproject.toml: \"pylibsparseir{pyproject_version}\"")
    if meta_yaml_version:
        print(f"  .conda/meta.yaml: \"spm-lab::pylibsparseir {meta_yaml_version}\"")
    print()
    
    # Check consistency
    if pyproject_normalized == meta_yaml_normalized:
        print("✅ Version specifications are consistent!")
        return True
    else:
        print("❌ Version specifications are inconsistent!")
        print(f"   pyproject.toml: {pyproject_normalized}")
        print(f"   .conda/meta.yaml: {meta_yaml_normalized}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("pylibsparseir Version Consistency Checker")
    print("=" * 60)
    print()
    
    is_consistent = check_version_consistency()
    
    print()
    print("=" * 60)
    if is_consistent:
        print("✅ All checks passed!")
        sys.exit(0)
    else:
        print("❌ Version inconsistency detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
