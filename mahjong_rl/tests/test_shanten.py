from mahjong.tile import TilesConverter as T

from mahjong_rl.engine.shanten import shanten, waits_counts
from mahjong_rl.engine.tiles import counts34


def sh(**k):
    return shanten(T.string_to_136_array(**k))


def test_complete_hand_is_minus_one():
    assert sh(man="123456789", pin="12355") == -1


def test_tenpai_is_zero():
    assert sh(man="123456789", pin="12", sou="55") == 0


def test_one_shanten():
    assert sh(man="123456789", pin="13", sou="59") == 1


def test_chiitoitsu_tenpai():
    assert sh(man="1133", pin="5599", sou="2244", honors="1") == 0


def test_kokushi_tenpai():
    assert sh(man="19", pin="19", sou="19", honors="1234567") == 0


def test_scattered_hand():
    # 13 isolated tiles: chiitoitsu path gives 6, kokushi 7, standard 8 -> best is 6
    assert sh(man="147", pin="258", sou="369", honors="1234") == 6


def test_waits():
    c = counts34(T.string_to_136_array(man="123456789", pin="12", sou="55"))
    assert waits_counts(c) == [11]  # 3p only
