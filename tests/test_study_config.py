import unittest

from fair_mia.study_config import load_study_config, resolve_experiments, stable_hash
from fair_mia.sweep import build_execution_plan


class StudyConfigTests(unittest.TestCase):
    def test_repository_yaml_resolves_without_pyyaml(self):
        config = load_study_config("configs/lora_studies.yaml")
        experiments = resolve_experiments(config, study_name="smoke_olmo_lora")
        self.assertEqual(len(experiments), 1)
        plan = build_execution_plan(config, experiments)
        self.assertEqual(plan["training_job_count"], 1)
        self.assertEqual(plan["scoring_job_count"], 2)
        self.assertEqual({job["stage"] for job in plan["jobs"]}, {
            "dataset_preparation",
            "training",
            "primitive_scoring",
            "attack_evaluation",
            "metrics_reporting",
        })

    def test_controlled_distribution_study_has_18_experiments(self):
        config = load_study_config("configs/lora_studies.yaml")
        experiments = resolve_experiments(
            config,
            study_name="ablation_training_distribution_pan17",
        )
        self.assertEqual(len(experiments), 18)
        plan = build_execution_plan(config, experiments)
        self.assertEqual(plan["training_job_count"], 9)
        self.assertEqual(plan["scoring_job_count"], 36)

    def test_stable_hash_ignores_mapping_order(self):
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))

    def test_matching_training_jobs_are_reused_and_gpu_plans_are_bounded(self):
        config = load_study_config("configs/lora_studies.yaml")
        plan = build_execution_plan(config, resolve_experiments(config))
        matching = [
            job
            for job in plan["jobs"]
            if job["stage"] == "training"
            and job["model"] == "qwen3_4b"
            and job["dataset"] == "pan17_multilingual"
            and job["training_variant"] == "balanced"
            and job["seed"] == 29
            and job["epochs"] == 4
        ]
        self.assertEqual(len(matching), 1)
        gpu_jobs = [
            job for job in plan["jobs"] if job["stage"] in {"training", "primitive_scoring"}
        ]
        self.assertTrue(gpu_jobs)
        self.assertTrue({job["planned_gpu"] for job in gpu_jobs}.issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
