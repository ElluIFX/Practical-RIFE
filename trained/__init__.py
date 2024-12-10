import importlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trained.V426H.RIFE_HDv3 import Model

MODEL_LIST = [
    f"{x[0:2]}.{x[2:]}"
    for x in os.listdir(os.path.dirname(__file__))
    if x.startswith("V") and os.path.isdir(os.path.join(os.path.dirname(__file__), x))
]
DEFAULT_MODEL = "V4.26"


def load_model(
    version: str,
) -> "Model":
    model: "Model" = importlib.import_module(
        f".{version.replace('.', '')}.RIFE_HDv3", package="trained"
    ).Model()
    model.load_model(
        os.path.join(os.path.dirname(__file__), version.replace(".", "")), -1
    )
    model.eval()
    model.device()
    return model
