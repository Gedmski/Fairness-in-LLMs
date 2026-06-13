import unittest

from fair_mia.finetune import _select_stratified_training_samples
from fair_mia.types import TextSample


class FineTuneSelectionTests(unittest.TestCase):
    def test_same_seed_returns_same_subset(self):
        samples = self._build_samples()
        left = _select_stratified_training_samples(samples, max_train_samples=4, seed=3)
        right = _select_stratified_training_samples(samples, max_train_samples=4, seed=3)
        self.assertEqual([sample.sample_id for sample in left], [sample.sample_id for sample in right])

    def test_different_seed_returns_different_subset(self):
        samples = self._build_samples()
        left = _select_stratified_training_samples(samples, max_train_samples=4, seed=1)
        right = _select_stratified_training_samples(samples, max_train_samples=4, seed=9)
        self.assertNotEqual([sample.sample_id for sample in left], [sample.sample_id for sample in right])

    def test_subset_stays_balanced_by_group_and_variety(self):
        samples = self._build_samples()
        selected = _select_stratified_training_samples(samples, max_train_samples=8, seed=5)
        counts: dict[tuple[str, str], int] = {}
        for sample in selected:
            key = (sample.group, str(sample.metadata["variety"]))
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(selected), 8)
        self.assertEqual(set(counts.values()), {2})

    def _build_samples(self) -> list[TextSample]:
        samples: list[TextSample] = []
        for group in ["G0", "G1"]:
            for variety in ["canada", "us"]:
                for index in range(4):
                    samples.append(
                        TextSample(
                            sample_id=f"{group}-{variety}-{index}",
                            text=f"{group} {variety} {index}",
                            is_member=True,
                            group=group,
                            scenario="finetuning",
                            metadata={"variety": variety},
                        )
                    )
        return samples


if __name__ == "__main__":
    unittest.main()
