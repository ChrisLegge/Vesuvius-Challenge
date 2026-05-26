"""
Unit tests for src/vesuvius/metrics.py.

Tests cover SurfaceDice@tau and VOI split/merge across the edge cases
most likely to break in practice: empty predictions, perfect overlap,
single-voxel surfaces, bridges, and fragmentation events.
"""

import numpy as np
import pytest
from vesuvius import surface_dice_tau, voi_split_merge, voi_score, composite_score


# ---------------------------------------------------------------------------
# SurfaceDice@tau
# ---------------------------------------------------------------------------

class TestSurfaceDiceTau:

    def test_both_empty_returns_one(self):
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt   = np.zeros((10, 10), dtype=np.uint8)
        assert surface_dice_tau(pred, gt) == 1.0

    def test_pred_empty_returns_zero(self):
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt   = np.zeros((10, 10), dtype=np.uint8)
        gt[4:6, 4:6] = 1
        assert surface_dice_tau(pred, gt) == 0.0

    def test_gt_empty_returns_zero(self):
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[4:6, 4:6] = 1
        gt = np.zeros((10, 10), dtype=np.uint8)
        assert surface_dice_tau(pred, gt) == 0.0

    def test_perfect_overlap_returns_one(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        score = surface_dice_tau(mask, mask, tau=2.0)
        assert score == pytest.approx(1.0)

    def test_score_increases_with_tau(self):
        """Wider tolerance should never decrease the score."""
        pred = np.zeros((30, 30), dtype=np.uint8)
        gt   = np.zeros((30, 30), dtype=np.uint8)
        pred[5:25, 10:20] = 1
        gt[6:25,   10:20] = 1   # one-row offset
        score_tight = surface_dice_tau(pred, gt, tau=0.5)
        score_wide  = surface_dice_tau(pred, gt, tau=3.0)
        assert score_wide >= score_tight

    def test_nearby_surfaces_score_higher_than_distant(self):
        """Surface 1 voxel away should score higher than surface 5 voxels away."""
        pred = np.zeros((40, 40), dtype=np.uint8)
        pred[10:30, 2:4] = 1

        gt_near = np.zeros_like(pred)
        gt_near[10:30, 3:5] = 1   # 1-voxel offset

        gt_far = np.zeros_like(pred)
        gt_far[10:30, 8:10] = 1   # 6-voxel offset

        score_near = surface_dice_tau(pred, gt_near, tau=2.0)
        score_far  = surface_dice_tau(pred, gt_far,  tau=2.0)
        assert score_near > score_far

    def test_spacing_scales_physical_distance(self):
        """Doubling voxel spacing should halve the effective tolerance."""
        pred = np.zeros((30, 30), dtype=np.uint8)
        gt   = np.zeros((30, 30), dtype=np.uint8)
        pred[5:25, 5:7] = 1
        gt[5:25, 8:10]  = 1   # 3-voxel gap in image space

        # tau=4 at unit spacing — gap of 3 is within tolerance
        score_unit   = surface_dice_tau(pred, gt, tau=4.0, spacing=(1.0, 1.0))
        # tau=4 at spacing=2 — physical gap is 6, outside tolerance
        score_coarse = surface_dice_tau(pred, gt, tau=4.0, spacing=(2.0, 2.0))
        assert score_unit > score_coarse


# ---------------------------------------------------------------------------
# VOI split / merge
# ---------------------------------------------------------------------------

class TestVOISplitMerge:

    def test_both_empty_returns_zero(self):
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt   = np.zeros((10, 10), dtype=np.uint8)
        split, merge = voi_split_merge(pred, gt)
        assert split == 0.0
        assert merge == 0.0

    def test_perfect_overlap_returns_zero(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        split, merge = voi_split_merge(mask, mask)
        assert split == pytest.approx(0.0, abs=1e-6)
        assert merge == pytest.approx(0.0, abs=1e-6)

    def test_bridge_raises_voi_split(self):
        """
        A prediction that bridges two separate GT surfaces produces a coarser
        labelling than GT. One pred component now covers two GT components, so
        H(GT | Pred) — returned as voi_split — increases.

        voi_merge = H(Pred | GT) stays zero: each GT component still maps
        entirely to the single merged pred component.
        """
        gt = np.zeros((30, 30), dtype=np.uint8)
        gt[5:10,  5:25] = 1   # surface 1
        gt[20:25, 5:25] = 1   # surface 2 (separate)

        pred_bridge = gt.copy()
        pred_bridge[5:25, 14:16] = 1  # neck connecting them

        split_clean,  _ = voi_split_merge(gt.copy(), gt)
        split_bridge, _ = voi_split_merge(pred_bridge, gt)
        assert split_bridge > split_clean

    def test_fragmentation_raises_voi_merge(self):
        """
        A prediction that fragments one GT surface into two pieces produces a
        finer labelling than GT. One GT component now maps to two pred
        components, so H(Pred | GT) — returned as voi_merge — increases.

        voi_split = H(GT | Pred) stays zero: each pred component still maps
        entirely to the single GT component.
        """
        gt = np.zeros((30, 30), dtype=np.uint8)
        gt[10:20, 5:25] = 1

        pred_split = np.zeros_like(gt)
        pred_split[10:20, 5:12]  = 1   # left fragment
        pred_split[10:20, 14:25] = 1   # right fragment (gap in middle)

        _, merge_clean = voi_split_merge(gt.copy(), gt)
        _, merge_split = voi_split_merge(pred_split,  gt)
        assert merge_split > merge_clean

    def test_voi_score_in_unit_interval(self):
        pred = np.zeros((20, 20), dtype=np.uint8)
        gt   = np.zeros((20, 20), dtype=np.uint8)
        pred[5:15, 5:15] = 1
        gt[6:14, 6:14]   = 1
        v = voi_score(pred, gt)
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

class TestCompositeScore:

    def test_perfect_prediction_scores_near_zero_point_seven(self):
        """
        With T=0 (not provided), perfect D_tau and V gives 0.35 + 0.35 = 0.70.
        """
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        result = composite_score(mask, mask, topo_score=0.0)
        assert result["composite"] == pytest.approx(0.70, abs=0.01)

    def test_all_components_present(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        result = composite_score(mask, mask)
        for key in ("surfdice", "voi_split", "voi_merge", "voi", "topo", "composite"):
            assert key in result

    def test_bridge_lowers_composite_score(self):
        """Adding a bridge should lower the composite score."""
        gt = np.zeros((40, 40), dtype=np.uint8)
        gt[5:12,  5:35] = 1
        gt[28:35, 5:35] = 1

        result_clean  = composite_score(gt.copy(), gt)
        pred_bridge = gt.copy()
        pred_bridge[12:28, 19:21] = 1
        result_bridge = composite_score(pred_bridge, gt)

        assert result_bridge["composite"] < result_clean["composite"]
