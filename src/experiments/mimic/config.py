from experiments.mimic.config_loader import load_global_config

_cfg = load_global_config()

N_CONDITIONS: int = _cfg['n_conditions']
