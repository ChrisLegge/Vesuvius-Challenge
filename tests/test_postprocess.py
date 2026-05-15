from vesuvius import hysteresis_components


def test_hysteresis_keeps_weak_component_touching_strong_seed():
    probabilities = [
        [0.0, 0.0, 0.0],
        [0.0, 0.7, 0.4],
        [0.0, 0.0, 0.0],
    ]

    assert hysteresis_components(probabilities, low=0.3, high=0.6) == [
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]


def test_hysteresis_removes_weak_component_without_strong_seed():
    probabilities = [
        [0.4, 0.4, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.7, 0.0],
    ]

    assert hysteresis_components(probabilities, low=0.3, high=0.6) == [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ]
