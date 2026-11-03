from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from footnote.retrieval.fusion import reciprocal_rank_fusion


def test_scores_match_the_formula() -> None:
    fused = reciprocal_rank_fusion({"bm25": [10, 20, 30], "dense": [20, 40]}, k=60)
    scores = {r.id: r.score for r in fused}

    assert scores[20] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 in bm25, rank 1 in dense
    assert scores[10] == pytest.approx(1 / 61)
    assert scores[40] == pytest.approx(1 / 62)
    assert scores[30] == pytest.approx(1 / 63)
    assert [r.id for r in fused] == [20, 10, 40, 30]
    assert fused[0].ranks == {"bm25": 2, "dense": 1}


def test_ties_break_on_best_rank_then_id() -> None:
    # 1 and 2 score the same (rank 1 in one list, rank 2 in the other); so do 3 and 4.
    fused = reciprocal_rank_fusion({"a": [1, 2, 4], "b": [2, 1, 3]}, k=60)
    assert [r.id for r in fused] == [1, 2, 3, 4]


def test_k_controls_how_much_the_top_rank_dominates() -> None:
    # x is first in one list; y is third in two lists (1-4 are filler).
    x, y = 100, 200
    rankings = {"a": [x], "b": [1, 2, y], "c": [3, 4, y]}

    def order(k: int) -> list[int]:
        return [r.id for r in reciprocal_rank_fusion(rankings, k=k) if r.id in (x, y)]

    assert order(0) == [x, y]  # 1/1 = 1.0 beats 1/3 + 1/3
    assert order(60) == [y, x]  # 2/63 beats 1/61


def test_duplicates_within_one_ranking_count_once() -> None:
    fused = reciprocal_rank_fusion({"a": [7, 7, 8]}, k=60)
    assert {r.id: r.ranks["a"] for r in fused} == {7: 1, 8: 3}
    assert fused[0].score == pytest.approx(1 / 61)


def test_empty_and_invalid_inputs() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"a": []}) == []
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({"a": [1]}, k=-1)


_ranking = st.lists(st.integers(0, 40), max_size=20, unique=True)


@given(st.dictionaries(st.sampled_from("abcd"), _ranking, max_size=4))
def test_fusion_is_independent_of_input_order(rankings: dict[str, list[int]]) -> None:
    forward = reciprocal_rank_fusion(rankings)
    backward = reciprocal_rank_fusion(dict(reversed(list(rankings.items()))))
    assert [(r.id, r.score) for r in forward] == [(r.id, r.score) for r in backward]


@given(st.dictionaries(st.sampled_from("abcd"), _ranking, min_size=1, max_size=4))
def test_an_item_first_everywhere_wins(rankings: dict[str, list[int]]) -> None:
    champion = 999
    boosted = {name: [champion, *ranking] for name, ranking in rankings.items()}
    fused = reciprocal_rank_fusion(boosted)
    assert fused[0].id == champion
    assert all(rank == 1 for rank in fused[0].ranks.values())
    assert {r.id for r in fused} == {champion, *(i for r in rankings.values() for i in r)}
