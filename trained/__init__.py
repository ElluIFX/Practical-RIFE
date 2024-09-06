import importlib
import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from trained.V43.RIFE_HDv3 import Model as Model_43
    from trained.V47.RIFE_HDv3 import Model as Model_47
    from trained.V414.RIFE_HDv3 import Model as Model_414
    from trained.V414L.RIFE_HDv3 import Model as Model_414L
    from trained.V415.RIFE_HDv3 import Model as Model_415
    from trained.V415L.RIFE_HDv3 import Model as Model_415L
    from trained.V416LB.RIFE_HDv3 import Model as Model_416LB
    from trained.V417.RIFE_HDv3 import Model as Model_417
    from trained.V417L.RIFE_HDv3 import Model as Model_417L
    from trained.V418.RIFE_HDv3 import Model as Model_418
    from trained.V420.RIFE_HDv3 import Model as Model_420
    from trained.V422.RIFE_HDv3 import Model as Model_422
    from trained.V422L.RIFE_HDv3 import Model as Model_422L

    MODEL = Union[
        Model_43,
        Model_47,
        Model_414,
        Model_414L,
        Model_415,
        Model_415L,
        Model_416LB,
        Model_417,
        Model_417L,
        Model_418,
        Model_420,
        Model_422,
        Model_422L,
    ]

MODEL_LIST = [
    "V4.3",
    "V4.7",
    "V4.14",
    "V4.14L",
    "V4.15",
    "V4.15L",
    "V4.16LB",
    "V4.17",
    "V4.17L",
    "V4.18",
    "V4.20",
    "V4.22",
    "V4.22L",
]
DEFAULT_MODEL = "V4.22"


def load_model(
    version: str,
) -> "MODEL":
    Model = importlib.import_module(
        f".{version.replace('.', '')}.RIFE_HDv3", package="trained"
    ).Model
    model: "MODEL" = Model()
    model.load_model(
        os.path.join(os.path.dirname(__file__), version.replace(".", "")), -1
    )
    model.eval()
    model.device()
    return model
