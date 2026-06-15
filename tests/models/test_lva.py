# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the root directory of this source tree.

import unittest

import torch

from src.models.lva import (
    LatentVarianceAssessment,
    discounted_survival_targets,
    predictive_variance,
)


class TestLVAUtilities(unittest.TestCase):
    def test_discounted_survival_targets(self):
        targets = discounted_survival_targets(torch.tensor([3, -1]), episode_length=5, gamma=0.9)
        self.assertEqual(targets.shape, (2, 5))
        self.assertAlmostEqual(float(targets[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(targets[0, 3]), 0.0, places=5)
        self.assertAlmostEqual(float(targets[0, 4]), 0.0, places=5)
        self.assertTrue(torch.allclose(targets[1], torch.ones(5)))

    def test_predictive_variance_zero_for_identical_predictions(self):
        preds = torch.ones(2, 4, 3, 8)
        variance = predictive_variance(preds)
        self.assertTrue(torch.allclose(variance, torch.zeros(2)))


class TestLVAModel(unittest.TestCase):
    def setUp(self):
        self.model = LatentVarianceAssessment(
            latent_dim=16,
            action_dim=4,
            noise_dim=8,
            dynamics_hidden_dim=32,
            perturb_hidden_dim=32,
            stability_hidden_dim=32,
        )

    def test_dynamics_loss_token_latents(self):
        z = torch.randn(3, 5, 16)
        action = torch.randn(3, 4)
        z_next = torch.randn(3, 5, 16)
        loss, pred = self.model.dynamics_loss(z, action, z_next)
        self.assertEqual(pred.shape, z_next.shape)
        self.assertEqual(loss.ndim, 0)

    def test_calibration_loss_and_variance_shape(self):
        z = torch.randn(3, 5, 16)
        action = torch.randn(3, 4)
        stability = torch.tensor([1.0, 0.5, 0.0])
        loss, variance, preds = self.model.calibration_loss(z, action, stability, perturbation_count=6)
        self.assertEqual(variance.shape, (3,))
        self.assertEqual(preds.shape, (3, 6, 5, 16))
        self.assertEqual(loss.ndim, 0)

    def test_stability_head_shape(self):
        z = torch.randn(3, 5, 16)
        scores = self.model.stability_head(z)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(torch.all((scores >= 0.0) & (scores <= 1.0)))


if __name__ == "__main__":
    unittest.main()
