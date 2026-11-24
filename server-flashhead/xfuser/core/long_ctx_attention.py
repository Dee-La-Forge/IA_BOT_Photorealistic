class xFuserLongContextAttention:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Multi-GPU inference is not available in this single-GPU setup.")
