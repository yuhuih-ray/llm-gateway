from llm_gateway.router_features import FEATURE_NAMES, extract_features


def test_pure_features_are_stable_and_handle_empty_text():
    assert extract_features("") == [0.0] * len(FEATURE_NAMES)
    question = r"WHY calculate $x+2$? Use `x` and \frac{1}{2}."
    assert extract_features(question) == extract_features(question)
    values = dict(zip(FEATURE_NAMES, extract_features(question)))
    assert values["latex_markers"] == 3
    assert values["code_markers"] == 6
    assert values["why"] == 1
    assert values["calculate"] == 1
    assert 0 < values["uppercase_ratio"] < 1
