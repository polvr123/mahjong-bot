"""Engine tests: full random hands plus scripted rule situations (ron, furiten, riichi, draws)."""
import random

import pytest

from mahjong_rl.engine import actions as A
from mahjong_rl.engine.game import IllegalAction, MahjongGame
from mahjong_rl.engine.state import PHASE_DONE, PHASE_RON, PHASE_TURN
from mahjong_rl.tests.helpers import JUNK_A, JUNK_B, JUNK_C, TENPAI_3P, scripted_game

DISCARD_3P = A.DISCARD_BASE + 11


# ---------------------------------------------------------------- random full hands
def test_initial_deal():
    g = MahjongGame(dealer=2, seed=5)
    s = g.state
    assert [len(h) for h in s.hands] == [13, 13, 14, 13]  # dealer has already drawn
    assert s.current_player == 2 and s.phase == PHASE_TURN
    assert s.wall.remaining == 136 - 14 - 52 - 1
    assert s.seat_winds == [29, 30, 27, 28]  # dealer (seat 2) is East; seats wrap around


@pytest.mark.parametrize("seed", range(150))
def test_random_hand_terminates_zero_sum_and_conserves_tiles(seed):
    rng = random.Random(seed)
    g = MahjongGame(dealer=seed % 4, seed=seed)
    for _ in range(1000):  # a hand has at most ~70 draws + a few ron offers
        if g.done:
            break
        g.step(g.decision_player, rng.choice(g.legal_actions()))
    assert g.done and g.state.phase == PHASE_DONE
    assert sum(g.result.deltas) == 0
    s = g.state
    tiles = [t for h in s.hands for t in h] + [t for d in s.discards for t in d]
    assert len(tiles) == len(set(tiles)) == s.wall._pos  # nothing duplicated or lost


def test_many_random_hands_include_wins_when_shanten_is_greedy():
    """Random players almost never win, so also check some hands end in wins using a greedy-ish policy:
    always tsumo/ron if possible, otherwise take riichi if available."""
    wins = 0
    for seed in range(60):
        rng = random.Random(seed)
        g = MahjongGame(dealer=seed % 4, seed=seed)
        while not g.done:
            acts = g.legal_actions()
            pick = next((a for a in (A.TSUMO, A.RON) if a in acts), None)
            if pick is None:
                riichi = [a for a in acts if A.RIICHI_BASE <= a < A.TSUMO]
                pick = rng.choice(riichi or acts)
            g.step(g.decision_player, pick)
        wins += g.result.kind != "draw"
        assert sum(g.result.deltas) == 0
    assert wins > 0


# ---------------------------------------------------------------- action validation
def test_illegal_actions_raise():
    g = MahjongGame(dealer=0, seed=1)
    with pytest.raises(IllegalAction):
        g.step(1, g.legal_actions()[0])       # not seat 1's turn
    with pytest.raises(IllegalAction):
        g.step(0, A.PASS)                     # nothing to pass on
    with pytest.raises(IllegalAction):
        g.step(0, A.TSUMO)                    # no winning hand on the first draw


# ---------------------------------------------------------------- ron
def ron_setup(p1_riichi=True, p1_discards=None):
    """Seat 0 holds junk + a 3p; seat 1 is tenpai on 3p. Seat 0 to act."""
    g = scripted_game([JUNK_A, TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(pin="3"))
    g.state.riichi_declared[1] = p1_riichi
    if p1_discards:
        g.state.discards[1] = p1_discards
    return g


def test_ron_offered_and_scored():
    g = ron_setup()
    g.step(0, DISCARD_3P)
    s = g.state
    assert s.phase == PHASE_RON and s.decision_player == 1
    assert set(g.legal_actions()) == {A.RON, A.PASS}
    g.step(1, A.RON)
    r = g.result
    assert (r.kind, r.winner, r.loser) == ("ron", 1, 0)
    assert "Riichi" in r.yaku
    assert r.deltas[1] == r.points and r.deltas[0] == -r.points and sum(r.deltas) == 0


def test_no_yaku_no_ron():
    g = ron_setup(p1_riichi=False)  # TENPAI_3P is complete on 3p but has no yaku
    g.step(0, DISCARD_3P)
    assert g.state.phase == PHASE_TURN and g.state.current_player == 1  # no offer; seat 1 just drew


def test_ippatsu_yaku_on_ron():
    g = ron_setup()
    g.state.ippatsu[1] = True
    g.step(0, DISCARD_3P)
    g.step(1, A.RON)
    assert "Ippatsu" in g.result.yaku


def test_furiten_from_own_discards_blocks_ron():
    g = ron_setup(p1_discards=[47])  # seat 1 once discarded a 3p (tile 47 is a 3p)
    assert g.state.is_furiten(1)
    g.step(0, DISCARD_3P)
    assert g.state.phase == PHASE_TURN and g.state.current_player == 1


def test_passing_a_ron_causes_temporary_furiten():
    g = ron_setup()
    g.step(0, DISCARD_3P)
    g.step(1, A.PASS)
    s = g.state
    assert s.furiten_temp[1] and s.is_furiten(1)
    assert s.phase == PHASE_TURN and s.current_player == 1  # game moves on to seat 1's draw


def test_head_bump_first_seat_after_discarder_wins():
    g = scripted_game(
        [JUNK_A, TENPAI_3P, dict(man="456789", pin="12", sou="23455"), JUNK_C], drawn=dict(pin="3")
    )
    g.state.riichi_declared[1] = g.state.riichi_declared[2] = True
    g.step(0, DISCARD_3P)
    assert g.state.ron_queue == [1, 2]
    g.step(1, A.PASS)
    assert g.state.decision_player == 2
    g.step(2, A.RON)
    assert g.result.winner == 2 and g.result.loser == 0


# ---------------------------------------------------------------- riichi
RIICHI_HAND = dict(man="123456789", pin="99", sou="13")  # tenpai, waits 2s


def test_riichi_offered_only_when_tenpai_after_discard():
    g = scripted_game([RIICHI_HAND, TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(honors="7"))
    acts = g.legal_actions()
    assert A.RIICHI_BASE + 33 in acts                     # discard the useless red dragon and riichi
    assert A.RIICHI_BASE + 18 not in acts                 # discarding 1s would break the hand
    g2 = scripted_game([JUNK_A, TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(honors="7"))
    assert not any(A.RIICHI_BASE <= a < A.TSUMO for a in g2.legal_actions())


def test_riichi_needs_enough_wall():
    g = scripted_game([RIICHI_HAND, TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(honors="7"))
    g.state.wall._pos = 136 - 14 - 3
    g._mask = None
    assert not any(A.RIICHI_BASE <= a < A.TSUMO for a in g.legal_actions())


def test_riichi_locks_the_hand():
    g = scripted_game([RIICHI_HAND, TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(honors="7"))
    g.step(0, A.RIICHI_BASE + 33)
    s = g.state
    assert s.riichi_declared[0] and s.ippatsu[0]
    # play the others out until seat 0 acts again
    while not g.done and g.decision_player != 0:
        g.step(g.decision_player, g.legal_actions()[0])
    if not g.done:
        acts = g.legal_actions()
        discards = [a for a in acts if a < A.TSUMO]
        assert discards == [A.DISCARD_BASE + g.state.drawn_tile // 4]  # only the drawn tile may go


def test_riichi_stick_is_refunded_if_riichi_discard_is_ronned():
    g = scripted_game([RIICHI_HAND, TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(pin="3"))
    g.state.riichi_declared[1] = True
    g.step(0, A.RIICHI_BASE + 11)  # riichi, discarding 3p, which seat 1 wins on
    g.step(1, A.RON)
    r = g.result
    assert not g.state.riichi_paid[0]
    assert r.deltas[0] == -r.points and r.deltas[1] == r.points  # no 1000-point stick involved


def test_riichi_stick_paid_when_someone_else_wins_later():
    g = scripted_game([RIICHI_HAND, TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(honors="7"))
    g.step(0, A.RIICHI_BASE + 33)
    assert g.state.riichi_paid[0]  # discard went unclaimed, stick is on the table


# ---------------------------------------------------------------- tsumo / draw
def tsumo_game():
    g = scripted_game([JUNK_A, TENPAI_3P, JUNK_B, JUNK_C], mover=1, drawn=dict(pin="3"))
    g.state.riichi_declared[1] = True
    return g


def test_tsumo_available_and_scored():
    g = tsumo_game()
    g.state.discards[1] = [0]  # seat 1 has discarded before, so this isn't its first draw (no chiihou)
    assert A.TSUMO in g.legal_actions()
    g.step(1, A.TSUMO)
    r = g.result
    assert r.kind == "tsumo" and r.winner == 1 and sum(r.deltas) == 0
    assert "Riichi" in r.yaku and "Menzen Tsumo" in r.yaku


def test_non_dealer_tsumo_on_first_draw_is_chiihou():
    g = tsumo_game()  # seat 1 is not the dealer and this is its very first draw
    g.step(1, A.TSUMO)
    assert g.result.yaku == ["Chiihou"] and g.result.points == 32000


def test_exhaustive_draw_has_no_payments():
    g = scripted_game([JUNK_A, TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(honors="7"))
    g.state.wall._pos = 136 - 14 - 2  # two draws left
    g._mask = None
    while not g.done:
        # discard the drawn tile every time so nobody can improve
        drawn = g.state.drawn_tile
        g.step(g.decision_player, A.DISCARD_BASE + drawn // 4)
    assert g.result.kind == "draw" and g.result.deltas == [0, 0, 0, 0] and g.state.wall.is_empty()
