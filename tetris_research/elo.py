from __future__ import annotations


class EloRatings:
    def __init__(self, initial: float = 1500.0, k: float = 32.0):
        self.initial, self.k, self.ratings = initial, k, {}

    def rating(self, name: str) -> float:
        return self.ratings.get(name, self.initial)

    def update(self, a: str, b: str, score_a: float) -> dict[str, float]:
        old_a, old_b = self.rating(a), self.rating(b)
        expected_a = 1 / (1 + 10 ** ((old_b - old_a) / 400))
        new_a = old_a + self.k * (score_a - expected_a)
        new_b = old_b + self.k * ((1 - score_a) - (1 - expected_a))
        self.ratings[a], self.ratings[b] = new_a, new_b
        return {"a_before": old_a, "b_before": old_b, "a_after": new_a, "b_after": new_b,
                "score_a": score_a}
