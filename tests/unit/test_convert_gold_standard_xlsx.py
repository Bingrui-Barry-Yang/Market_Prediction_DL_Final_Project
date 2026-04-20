from scripts.convert_gold_standard_xlsx import integrated_gold_score


def test_integrated_gold_score_bands() -> None:
    assert integrated_gold_score("-1.0", "1.0") == 1
    assert integrated_gold_score("-1.0", "5.0") == 5
    assert integrated_gold_score("0.0", "1.0") == 6
    assert integrated_gold_score("0.0", "5.0") == 10
    assert integrated_gold_score("1.0", "1.0") == 11
    assert integrated_gold_score("1.0", "5.0") == 15
