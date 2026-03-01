import sys
import os
import json
import subprocess

def get_torch_include_paths():
    try:
        import torch
        from torch.utils.cpp_extension import include_paths
        return include_paths()
    except ImportError:
        print("Error: 'torch' is not installed in this environment.")
        print("Please install torch first (e.g., 'pip install torch').")
        sys.exit(1)

def update_vscode_config():
    paths = get_torch_include_paths()
    print(f"Found torch include paths: {paths}")
    
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vscode_dir = os.path.join(workspace_root, ".vscode")
    config_file = os.path.join(vscode_dir, "c_cpp_properties.json")
    
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir)
        
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing {config_file}. Creating new one.")
            
    if "configurations" not in config:
        config["configurations"] = []
        
    if not config["configurations"]:
        config["configurations"].append({
            "name": "Linux",
            "includePath": [],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "gnu++17",
            "intelliSenseMode": "linux-gcc-x64"
        })
        
    # Ensure all torch paths are in includePath
    for conf in config["configurations"]:
        if "includePath" not in conf:
            conf["includePath"] = ["${workspaceFolder}/**"]
            
        current_paths = set(conf["includePath"])
        for p in paths:
            if p not in current_paths:
                conf["includePath"].append(p)
                
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated {config_file} with torch include paths.")

if __name__ == "__main__":
    update_vscode_config()
