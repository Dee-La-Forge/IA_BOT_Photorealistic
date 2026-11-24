def _single_gpu_only(*args, **kwargs):
    raise RuntimeError("Multi-GPU inference is not available in this single-GPU setup.")


def get_sequence_parallel_rank():
    return 0


def get_sequence_parallel_world_size():
    return 1


get_sp_group = get_world_group = _single_gpu_only
init_distributed_environment = initialize_model_parallel = _single_gpu_only
