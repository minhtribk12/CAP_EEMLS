from __future__ import annotations

import random

from eemls.image_catalogue import Catalogue

IDS = {
    ("car", "day", "clear"): ("car-day-0", "car-day-1"),
    ("car", "night", "clear"): ("car-night-0",),
    ("truck", "day", "clear"): ("truck-day-0",),
    ("truck", "day", "blurred"): ("truck-blurred-0",),
}


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311


class TestADraw:
    def test_it_draws_a_clear_day_image_of_a_dominant_class(self) -> None:
        drawn = {
            Catalogue(IDS).draw(dominant_classes=("car", "truck"), draws=seeded(seed))
            for seed in range(20)
        }

        assert drawn <= {"car-day-0", "car-day-1", "truck-day-0"}

    def test_the_same_seed_draws_the_same_images(self) -> None:
        catalogue = Catalogue(IDS)
        first, second = seeded(3), seeded(3)

        assert [catalogue.draw(dominant_classes=("car",), draws=first) for _ in range(5)] == [
            catalogue.draw(dominant_classes=("car",), draws=second) for _ in range(5)
        ]

    def test_it_picks_the_class_first_and_then_the_image(self) -> None:
        catalogue, draws, expected = Catalogue(IDS), seeded(7), seeded(7)

        label = expected.choice(("car", "truck"))
        image = catalogue.pick(
            label=label, condition="day", image_quality="clear", draw=expected.random()
        )

        assert catalogue.draw(dominant_classes=("car", "truck"), draws=draws) == image
