import pydantic
from typing import Literal, List, Optional
from pydantic import BaseModel
import json


class Cable(BaseModel):
    start_node: str
    end_node: str
    cable_type: str

class Geometry(BaseModel):
    name: str
    clearance: float
    preference: Literal['Near', 'Avoid', 'Neutral']
    clipable: bool
    assembly: bool

class HarnessSetup(BaseModel):
    geometries: List[Geometry]
    cables:List[Cable]


def write_harness_setup_to_json(harness_setup: HarnessSetup, file_path: str) -> None:
    """
    Write a HarnessSetup instance to a JSON file.
    Args:
    - harness_setup (HarnessSetup): The instance to be written to a file.
    - file_path (str): The path to the file where the instance will be written.
    """
    with open(file_path, 'w') as file:
        json.dump(harness_setup.model_dump(), file, indent=4)

def load_harness_setup_from_json(file_path: str) -> Optional[HarnessSetup]:
    """
    Load a HarnessSetup instance from a JSON file.
    Args:
    - file_path (str): The path to the file from which the instance will be loaded.
    Returns:
    - HarnessSetup: The loaded instance.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return HarnessSetup(**data)
    except Exception as e:
        print(f"Error while loading HarnessSetup from {file_path}: {e}")
        return None