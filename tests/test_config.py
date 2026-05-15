from pathlib import Path

from vesuvius import load_inference_config


def test_inference_weights_sum_to_one():
    config = load_inference_config(Path("configs/inference_config_summary.json"))

    assert round(config.weight_sum, 4) == 1.0
    assert len(config.active_models) == 3
    assert config.degradation_levels == 6
