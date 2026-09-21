import random

from mahjong_rl.engine.tiles import NUM_TILES, Wall, counts34, dora_from_indicator, tile_name, tile_type


def test_wall_deal_and_counts():
    w = Wall(random.Random(0))
    assert sorted(w.tiles) == list(range(NUM_TILES))
    hands = w.deal()
    assert [len(h) for h in hands] == [13] * 4
    assert w.remaining == 136 - 14 - 52
    assert len(w.dead_wall) == 14
    assert len({t for h in hands for t in h}) == 52
    assert w.dora_indicator in w.dead_wall


def test_wall_draws_until_empty():
    w = Wall(random.Random(1))
    drawn = [w.draw() for _ in range(w.remaining)]
    assert w.is_empty()
    assert not set(drawn) & set(w.dead_wall)
    try:
        w.draw()
        assert False, "expected IndexError"
    except IndexError:
        pass


def test_tile_helpers():
    assert tile_type(0) == 0 and tile_type(135) == 33
    assert tile_name(0) == "1m" and tile_name(4 * 9) == "1p" and tile_name(4 * 26) == "9s" and tile_name(4 * 33) == "C"
    assert set(counts34(range(136))) == {4}


def test_dora_wraps():
    assert dora_from_indicator(8) == 0      # 9m -> 1m
    assert dora_from_indicator(17) == 9     # 9p -> 1p
    assert dora_from_indicator(0) == 1      # 1m -> 2m
    assert dora_from_indicator(30) == 27    # N -> E
    assert dora_from_indicator(33) == 31    # red dragon -> white
    assert dora_from_indicator(31) == 32
