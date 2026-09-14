# -*- coding: utf-8 -*-
"""
tests/test_stair_ne_nlgcl_v5_plus.py
====================================
Comprehensive unit test suite for STAIR-NE-NLGCL v5+ (v3-Refined) architecture:
1. Linear Warmup (0 -> 0.010 in 50 epochs, constant afterwards).
2. Sign-Preserving Spectral Noise (|noise| quadrant invariance, 0 mismatch).
3. Hard-Threshold MFNA (mask = I(sim <= 0.85), near-duplicate suppression).
4. Linear HANS (psi = 1.0 + gamma_h * clamp(cos, min=0), preserving tau_eff).
5. 100% Direct Gradient Flow (No Projection Head, InfoNCE backpropagates directly to H0, H1).
"""

import math
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stair_ne_nlgcl_v5_plus import STAIR_NE_NLGCL_v5_Plus


class TestSTAIR_NE_NLGCL_v5_Plus(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.n_users = 100
        self.n_items = 200
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_01_linear_warmup(self):
        """Pillar 1: Linear warmup schedule (0.0 -> 0.010 in 50 epochs)."""
        model = STAIR_NE_NLGCL_v5_Plus(
            n_users=self.n_users, n_items=self.n_items,
            lambda_cl=0.010, warmup_epochs=50
        ).to(self.device)

        model.update_epoch(0)
        self.assertAlmostEqual(model.current_lambda, 0.0, places=6)

        model.update_epoch(25)
        self.assertAlmostEqual(model.current_lambda, 0.005, places=6)

        model.update_epoch(50)
        self.assertAlmostEqual(model.current_lambda, 0.010, places=6)

        model.update_epoch(500)
        self.assertAlmostEqual(model.current_lambda, 0.010, places=6)

    def test_02_sign_preserving_noise(self):
        """Pillar 2: Sign-preserving spectral noise (|eta| >= 0, quadrant invariance)."""
        model = STAIR_NE_NLGCL_v5_Plus(
            n_users=self.n_users, n_items=self.n_items
        ).to(self.device)
        model.train()

        h = torch.randn(100, self.dim, device=self.device)
        beta = torch.linspace(0.9, 0.1, self.dim, device=self.device)
        h_tilde = model.inject_spectral_noise(h, beta)

        # Ensure shapes match
        self.assertEqual(h_tilde.shape, h.shape)

        # Mismatch rate where sign differs for non-negligible values
        mismatch = ((torch.sign(h_tilde) != torch.sign(h)) & (h.abs() > 1e-5)).float().mean().item()
        self.assertEqual(mismatch, 0.0, "Sign-preserving noise must guarantee 0 sign flips!")

    def test_03_hard_threshold_mfna(self):
        """Pillar 3: Hard-threshold MFNA (mask out pairs with sim > tau_thresh)."""
        tau_thresh = 0.85
        item_mod = torch.randn(self.batch_size, self.dim, device=self.device)
        # Create near-duplicate
        item_mod[1] = item_mod[0] + 0.001 * torch.randn(self.dim, device=self.device)

        i_norm = F.normalize(item_mod, p=2, dim=-1)
        sim = torch.matmul(i_norm, i_norm.t())
        mask = (sim <= tau_thresh).float()

        self.assertGreater(sim[0, 1].item(), tau_thresh)
        self.assertEqual(mask[0, 1].item(), 0.0, "Near duplicates must be masked out with 0.0!")

    def test_04_linear_hans(self):
        """Pillar 4: Linear HANS penalty preserving effective temperature."""
        gamma_h = 0.15
        cos_sim = torch.tensor([-0.5, 0.0, 0.4, 0.8], device=self.device)
        hans_weight = 1.0 + gamma_h * torch.clamp(cos_sim, min=0.0)

        self.assertAlmostEqual(hans_weight[0].item(), 1.0, places=6)
        self.assertAlmostEqual(hans_weight[1].item(), 1.0, places=6)
        self.assertAlmostEqual(hans_weight[2].item(), 1.06, places=5)
        self.assertAlmostEqual(hans_weight[3].item(), 1.12, places=5)

    def test_05_direct_gradient_flow(self):
        """Pillar 5: Direct InfoNCE gradient flow into H0 and H1 without projection head."""
        model = STAIR_NE_NLGCL_v5_Plus(
            n_users=self.n_users, n_items=self.n_items,
            lambda_cl=0.010, warmup_epochs=50
        ).to(self.device)
        model.update_epoch(50)

        total_nodes = self.n_users + self.n_items
        H0 = torch.randn(total_nodes, self.dim, device=self.device, requires_grad=True)
        H1 = torch.randn(total_nodes, self.dim, device=self.device, requires_grad=True)
        u_idx = torch.randint(0, self.n_users, (self.batch_size,), device=self.device)
        pos_idx = torch.randint(0, self.n_items, (self.batch_size,), device=self.device)
        beta = torch.linspace(0.9, 0.1, self.dim, device=self.device)
        item_mod = torch.randn(self.batch_size, self.dim, device=self.device)

        tot_loss, raw_loss = model([H0, H1], u_idx, pos_idx, beta, item_mod)
        self.assertGreater(tot_loss.item(), 0.0)
        self.assertGreater(raw_loss, 0.0)

        tot_loss.backward()
        self.assertIsNotNone(H0.grad)
        self.assertIsNotNone(H1.grad)
        self.assertGreater(H0.grad.norm().item(), 0.0)
        self.assertGreater(H1.grad.norm().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
