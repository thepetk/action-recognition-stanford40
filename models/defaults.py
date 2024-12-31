import os
from models.utils import ModelChoice

MODEL = os.getenv("MODEL", "")
SAVE_AS_YAML = bool(os.getenv("SAVE_AS_YAML", True))


def get_chosen_models(model: "str" = MODEL) -> "list[str]":
    if model == ModelChoice.CUSTOM:
        print("Model:: Only custom model is included")
        return [ModelChoice.CUSTOM]
    elif model == ModelChoice.RESNET:
        print("Model:: Only resnet model is included")
        return [ModelChoice.RESNET]
    else:
        print("Model:: Both models are included")
        return [ModelChoice.CUSTOM, ModelChoice.RESNET]


CHOSEN_MODELS = get_chosen_models()
