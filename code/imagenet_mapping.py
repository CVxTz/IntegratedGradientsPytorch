import json
from pathlib import Path
import os

with open(Path(os.path.abspath(__file__)).parent / 'mapping.json', 'r') as f:
    mapping = json.load(f)

mapping = {int(k): v for k, v in mapping.items()}

