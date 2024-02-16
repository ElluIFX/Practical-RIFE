import os


def get_model(version):
    if version == "V4.3":
        from .V43.RIFE_HDv3 import Model
    elif version == "V4.7":
        from .V47.RIFE_HDv3 import Model
    elif version == "V4.14":
        from .V414.RIFE_HDv3 import Model
    elif version == "V4.14L":
        from .V414L.RIFE_HDv3 import Model
    else:
        raise ImportError(f"Model {version} not available")
    model = Model()
    model.load_model(
        os.path.join(os.path.dirname(__file__), version.replace(".", "")), -1
    )
    model.eval()
    model.device()
    return model


model_list = ["V4.3", "V4.7", "V4.14", "V4.14L"]
