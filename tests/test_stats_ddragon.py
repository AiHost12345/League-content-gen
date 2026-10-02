import math

import pytest

from buildopt.stats import confidence_label, effective_n, logit, pair_lift, wilson_interval


def test_wilson_matches_reference_values():
    lo, hi = wilson_interval(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)
    lo, hi = wilson_interval(0.0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)


def test_effective_n():
    assert effective_n([1, 1, 1, 1]) == pytest.approx(4)
    assert effective_n([1, 0.3]) == pytest.approx(1.3 ** 2 / 1.09)


def test_pair_lift_is_zero_when_effects_add_on_log_odds():
    champ, a, b = 0.5, 0.55, 0.53
    ab = 1 / (1 + math.exp(-(logit(a) + logit(b) - logit(champ))))
    assert pair_lift(ab, a, b, champ) == pytest.approx(0.0, abs=1e-9)
    assert pair_lift(ab + 0.05, a, b, champ) > 0


def test_confidence_labels():
    assert confidence_label(120, 0.45, 0.50) == "low"  # under 200 games is always low
    assert confidence_label(5000, 0.48, 0.52) == "high"
    assert confidence_label(500, 0.40, 0.52) == "medium"


def test_completed_item_classification(static):
    assert static.is_legendary(6676)  # The Collector
    assert static.is_legendary(3071)  # Black Cleaver
    assert static.is_legendary(3748)  # Titanic Hydra
    assert not static.is_legendary(1036)  # Long Sword
    assert not static.is_legendary(2003)  # Health Potion
    assert static.is_boots(3047)  # Plated Steelcaps
    assert not static.is_boots(1001)  # basic Boots
    assert static.is_completed(3047) and not static.is_completed(1001)


def test_name_lookup(static):
    assert static.champion_id("Briar") == 233
    assert static.champion_id("kha'zix") == static.champion_id("Khazix")
    assert static.item_id("BC") == 3071
    assert static.item_id("Collector") == 6676
    assert static.item_id("titanic") == 3748
    assert static.item_id("Death's Dance") == 6333
    assert "anti-heal" in static.item_categories(6609)  # Chempunk Chainsword
    assert "armor" in static.item_categories(6333)
    assert static.keystones >= {8010, 8008, 8112}
