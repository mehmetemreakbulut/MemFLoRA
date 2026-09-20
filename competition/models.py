def build_model(name="mobilenet_v2", num_classes=17):
    """
    Build the frozen backbone
    """

def freeze_backbone(model):
    """
    Freeze all original backbone parameters.
    """

def get_target_layers(model):
    """
    Return the convolution layers where MemFLoRA adapters should be inserted.
    """