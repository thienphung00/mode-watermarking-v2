#!/usr/bin/env python3
"""Quick diagnostic script to check if a config file is valid."""

import sys
from pathlib import Path

def check_config_file(config_path: str):
    """Check if a config file exists and is valid."""
    path = Path(config_path)
    
    print(f"Checking: {config_path}")
    print(f"  Absolute path: {path.absolute()}")
    
    if not path.exists():
        print(f"  ❌ File does not exist!")
        return False
    
    file_size = path.stat().st_size
    print(f"  File size: {file_size} bytes")
    
    if file_size == 0:
        print(f"  ❌ File is empty!")
        return False
    
    # Read content
    with open(path, "r") as f:
        content = f.read()
    
    print(f"  Content length: {len(content)} chars")
    print(f"  First 100 chars: {repr(content[:100])}")
    
    # Try to parse YAML
    import yaml
    try:
        data = yaml.safe_load(content)
        print(f"  YAML parsed: ✅")
        print(f"  Parsed type: {type(data)}")
        
        if data is None:
            print(f"  ❌ YAML parsed to None (file might only contain comments)")
            return False
        
        if not isinstance(data, dict):
            print(f"  ❌ YAML root is not a dict, got: {type(data)}")
            return False
        
        print(f"  Keys in config: {list(data.keys())}")
        
        # Try to load as AppConfig
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from core.config import AppConfig
        try:
            app_config = AppConfig.from_yaml(config_path)
            print(f"  AppConfig loaded: ✅")
            print(f"  Watermark mode: {app_config.watermark.mode}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to load as AppConfig: {e}")
            return False
            
    except yaml.YAMLError as e:
        print(f"  ❌ Invalid YAML: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_config.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    success = check_config_file(config_path)
    sys.exit(0 if success else 1)

