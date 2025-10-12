

class ModelInitializer:
    @staticmethod
    def init_SBAINet(args, pre_trained=False):
        from model.SABPI_Net import SBAINet 
        model = SBAINet(in_chans=3,
                        patch_size=4,
                        window_size=7,
                        embed_dim=96,
                        depths=(2, 2, 6, 2),
                        num_heads=(3, 6, 12, 24),
                        num_classes=args.num_classes,
                        agent_num=[9, 16, 49, 49],
                        resolution=[28, 14, 7, 7],
                        attn_type='1100')
        return model

# Map modeling names to their respective initializer methods
MODEL_INITIALIZERS = {
    'SBAINet': ModelInitializer.init_SBAINet
}

def getModel(args, pre_trained=True):
    """
    Get a modeling instance based on the modeling name.
    Models are loaded lazily - only the requested modeling's code is imported.

    Args:
        args: Arguments containing model_name and other parameters
        pre_trained: Whether to load pre-trained weights

    Returns:
        Initialized modeling
    """
    if args.model_name in MODEL_INITIALIZERS:
        # Get the initializer function for the requested modeling
        initializer = MODEL_INITIALIZERS[args.model_name]
        return initializer(args, pre_trained)
    else:
        raise ValueError(f"Model '{args.model_name}' not found in the available models.")
