import os
from pathlib import Path

def ensure_project_structure():
    """Ensure required directories exist"""
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "data",
        "models",
        "checkpoints"
    ]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
    
    return base_dir