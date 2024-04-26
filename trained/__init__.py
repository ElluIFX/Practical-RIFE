import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from trained.V43.RIFE_HDv3 import Model as Model_43
    from trained.V47.RIFE_HDv3 import Model as Model_47
    from trained.V414.RIFE_HDv3 import Model as Model_414
    from trained.V414L.RIFE_HDv3 import Model as Model_414L
    from trained.V415.RIFE_HDv3 import Model as Model_415
    from trained.V415L.RIFE_HDv3 import Model as Model_415L

    MODEL = Union[Model_43, Model_47, Model_414, Model_414L, Model_415, Model_415L]

MODEL_LIST = ["V4.3", "V4.7", "V4.14", "V4.14L", "V4.15", "V4.15L"]
DEFAULT_MODEL = "V4.15"


def load_model(
    version: str,
) -> "MODEL":
    if version == "V4.3":
        from .V43.RIFE_HDv3 import Model
    elif version == "V4.7":
        from .V47.RIFE_HDv3 import Model
    elif version == "V4.14":
        from .V414.RIFE_HDv3 import Model
    elif version == "V4.14L":
        from .V414L.RIFE_HDv3 import Model
    elif version == "V4.15":
        from .V415.RIFE_HDv3 import Model
    elif version == "V4.15L":
        from .V415L.RIFE_HDv3 import Model
    else:
        raise ImportError(f"Model {version} not available")
    model = Model()
    model.load_model(
        os.path.join(os.path.dirname(__file__), version.replace(".", "")), -1
    )
    model.eval()
    model.device()
    return model
