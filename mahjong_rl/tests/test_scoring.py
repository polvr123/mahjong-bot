"""Scoring integration against known hands. Expected values follow the standard han/fu payment
tables and were cross-checked against the `mahjong` library; there is no yaku logic in our code."""
from mahjong.tile import TilesConverter as T

from mahjong_rl.engine.scoring import payments, score_hand

E, S = 27, 28


def h(**k):
    return T.string_to_136_array(**k)


def score(tiles, win, **kw):
    kw.setdefault("player_wind", S)
    kw.setdefault("round_wind", E)
    kw.setdefault("is_tsumo", False)
    return score_hand(tiles, win, **kw)


PINFU_SANSHOKU = h(man="234567", pin="23455", sou="234")
WIN_4S = h(sou="4")[0]


def test_pinfu_riichi_tsumo_tanyao_sanshoku_haneman():
    r = score(PINFU_SANSHOKU, WIN_4S, is_tsumo=True, is_riichi=True)
    assert r.is_valid_win and (r.han, r.fu, r.points) == (6, 20, 12000)
    assert set(r.yaku) == {"Menzen Tsumo", "Riichi", "Pinfu", "Tanyao", "Sanshoku Doujun"}
    assert (r.cost["main"], r.cost["additional"]) == (6000, 3000)


def test_dora_adds_han():
    dora_ind = h(man="1")[0]  # indicator 1m -> dora 2m, hand holds one 2m
    r = score(PINFU_SANSHOKU, WIN_4S, is_tsumo=True, is_riichi=True, dora_indicators=[dora_ind])
    assert r.han == 7 and "Dora 1" in r.yaku


def test_ippatsu_haitei_houtei_flags():
    assert "Ippatsu" in score(PINFU_SANSHOKU, WIN_4S, is_tsumo=True, is_riichi=True, is_ippatsu=True).yaku
    assert "Haitei Raoyue" in score(PINFU_SANSHOKU, WIN_4S, is_tsumo=True, is_haitei=True).yaku
    assert "Houtei Raoyui" in score(PINFU_SANSHOKU, WIN_4S, is_riichi=True, is_houtei=True).yaku


def test_riichi_tanyao_dora_dealer_ron():
    tiles = h(man="234555", pin="22", sou="345678")
    r = score(tiles, h(sou="6")[0], is_riichi=True, player_wind=E, dora_indicators=[h(man="1")[0]])
    assert (r.han, r.fu, r.points) == (3, 40, 7700)  # 3 han 40 fu dealer ron


def test_chiitoitsu_25_fu():
    tiles = h(man="1133", pin="5599", sou="2244", honors="11")
    r = score(tiles, tiles[-1])
    assert (r.han, r.fu, r.points, r.yaku) == (2, 25, 1600, ["Chiitoitsu"])


def test_kokushi_yakuman():
    tiles = h(man="119", pin="19", sou="19", honors="123456") + h(honors="7")
    r = score(tiles, h(honors="7")[0])
    assert r.is_valid_win and r.points == 32000 and r.yaku == ["Kokushi Musou"]


def test_suuankou_tsumo_yakuman_but_ron_is_not():
    tiles = h(man="111", pin="22299", sou="333444")
    tsumo = score(tiles, tiles[-1], is_tsumo=True)
    assert tsumo.points == 32000 and tsumo.yaku == ["Suu Ankou"]
    assert (tsumo.cost["main"], tsumo.cost["additional"]) == (16000, 8000)
    ron = score(tiles, tiles[-1])  # shanpon ron -> the completed triplet counts as open
    assert ron.points == 8000 and set(ron.yaku) == {"Toitoi", "San Ankou"}


def test_yakuhai_haku_1han_40fu():
    tiles = h(man="123", pin="456", sou="789", honors="555") + h(pin="99")
    r = score(tiles, h(man="3")[0])
    assert (r.han, r.fu, r.points) == (1, 40, 1300) and r.yaku == ["Yakuhai (haku)"]


def test_double_east_dealer_tsumo_and_payments():
    tiles = h(man="12355", pin="456", sou="789", honors="111")
    r = score(tiles, h(man="3")[0], is_tsumo=True, player_wind=E)
    assert (r.han, r.fu, r.points) == (3, 40, 7800)  # 2600 all
    assert set(r.yaku) == {"Menzen Tsumo", "Yakuhai (seat wind east)", "Yakuhai (round wind east)"}
    assert payments(r, winner=0, dealer=0, loser=None) == [7800, -2600, -2600, -2600]


def test_nondealer_tsumo_and_ron_payments():
    r = score(PINFU_SANSHOKU, WIN_4S, is_tsumo=True, is_riichi=True)
    assert payments(r, winner=1, dealer=0, loser=None) == [-6000, 12000, -3000, -3000]
    ron = score(PINFU_SANSHOKU, WIN_4S, is_riichi=True)
    assert payments(ron, winner=1, dealer=0, loser=3) == [0, ron.points, 0, -ron.points]


def test_complete_hand_without_yaku_is_not_a_win():
    tiles = h(man="123", pin="234456", sou="78999")
    r = score(tiles, h(pin="3")[0])
    assert not r.is_valid_win and r.error == "no_yaku"


def test_non_winning_hand():
    tiles = h(man="1379", pin="1379", sou="13579", honors="1")
    r = score(tiles, tiles[-1], is_tsumo=True)
    assert not r.is_valid_win and r.as_tuple() == (False, 0, 0, 0, [])
