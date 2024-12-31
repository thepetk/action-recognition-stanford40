class ModelChoice:
    """
    Covers the model decision between pretrained or custom NN
    """

    RESNET = "pretrained"
    CUSTOM = "custom"


class ModelOperation:
    """
    Defines the type of operation that the model manager will perform
    """

    TRAIN = 0
    VALIDATE = 1
    TEST = 2
