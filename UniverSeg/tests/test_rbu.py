import sys
import random
from pathlib import Path

import torch
try:
    import pytest
except ImportError:
    pytest = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT.parent / "UniverSeg-main"

sys.path.insert(0, str(OFFICIAL_REPO))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from universeg import universeg
from training.augmentations import apply_recipe_r_augmentations
from training.losses import soft_dice_loss


def test_zero_init_equivalence_with_baseline():
    """
    Rule 2 & 4: With --use_proto ON and zero-init extra channels, the model
    must give the exact same output as the baseline for the same weights and inputs.
    """
    torch.manual_seed(0)
    m_base = universeg(pretrained=False, use_proto=False).eval()

    torch.manual_seed(0)
    m_proto = universeg(pretrained=False, use_proto=True).eval()

    q = torch.rand(1, 1, 128, 128)
    s_img = torch.rand(1, 4, 1, 128, 128)
    s_msk = (torch.rand(1, 4, 1, 128, 128) > 0.5).float()

    with torch.no_grad():
        out_base = m_base(q, s_img, s_msk)
        out_proto = m_proto(q, s_img, s_msk)

    assert torch.allclose(out_base, out_proto, atol=1e-6), (
        f"Max diff: {(out_base - out_proto).abs().max().item()}"
    )


def test_permutation_invariance_over_s():
    """
    Rule 5: Keep S variable and the model permutation-invariant over S.
    Even after non-zero weights are placed on the prototype channels, permuting
    the support examples along dim=1 must leave the query prediction unchanged.
    """
    torch.manual_seed(42)
    m_proto = universeg(pretrained=False, use_proto=True).eval()

    # Give non-zero random weights to the 2 prototype channels so prototype branch is active
    with torch.no_grad():
        for block in list(m_proto.enc_blocks) + list(m_proto.dec_blocks):
            block.cross.cross_conv.weight[:, -2:, :, :].normal_(0.0, 0.1)

    q = torch.rand(1, 1, 128, 128)
    s_img = torch.rand(1, 6, 1, 128, 128)
    s_msk = (torch.rand(1, 6, 1, 128, 128) > 0.5).float()

    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    s_img_perm = s_img[:, perm]
    s_msk_perm = s_msk[:, perm]

    with torch.no_grad():
        out_1 = m_proto(q, s_img, s_msk)
        out_2 = m_proto(q, s_img_perm, s_msk_perm)

    assert torch.allclose(out_1, out_2, atol=1e-5), (
        f"Permutation invariance violated! Max diff: {(out_1 - out_2).abs().max().item()}"
    )


def test_variable_support_sizes_and_empty_masks(support_size=4):
    """
    Part 2: Test shapes and numerical stability for S in {1, 4, 8, 64},
    including when a support mask is completely empty (all 0s).
    """
    torch.manual_seed(7)
    m_proto = universeg(pretrained=False, use_proto=True).eval()

    q = torch.rand(1, 1, 128, 128)
    s_img = torch.rand(1, support_size, 1, 128, 128)
    s_msk = (torch.rand(1, support_size, 1, 128, 128) > 0.5).float()
    # Force first support mask to be completely empty to test safe handling
    s_msk[:, 0] = 0.0

    with torch.no_grad():
        out = m_proto(q, s_img, s_msk)

    assert out.shape == (1, 1, 128, 128)
    assert not torch.isnan(out).any(), "NaN detected in output!"
    assert not torch.isinf(out).any(), "Inf detected in output!"


def test_recipe_r_augmentations_binary_masks_and_range():
    """
    Part 1: Verify Recipe R augmentations keep masks strictly binary {0, 1}
    and images clamped to [0, 1].
    """
    rng = random.Random(123)
    q_img = torch.rand(1, 128, 128)
    q_msk = (torch.rand(1, 128, 128) > 0.5).float()
    s_img = torch.rand(4, 1, 128, 128)
    s_msk = (torch.rand(4, 1, 128, 128) > 0.5).float()

    for _ in range(10):
        qi, qm, si, sm = apply_recipe_r_augmentations(q_img, q_msk, s_img, s_msk, rng)
        assert qi.shape == (1, 128, 128)
        assert qm.shape == (1, 128, 128)
        assert si.shape == (4, 1, 128, 128)
        assert sm.shape == (4, 1, 128, 128)
        assert qi.min() >= 0.0 and qi.max() <= 1.0
        assert si.min() >= 0.0 and si.max() <= 1.0
        assert set( torch.unique(qm).tolist() ).issubset({0.0, 1.0})
        assert set( torch.unique(sm).tolist() ).issubset({0.0, 1.0})


def test_one_training_step_proto_gradients():
    """
    Rule 8: Verify one training step with --use_proto ON updates the
    prototype weights cleanly without NaNs.
    """
    torch.manual_seed(0)
    m_proto = universeg(pretrained=False, use_proto=True).train()
    opt = torch.optim.Adam(m_proto.parameters(), lr=1e-4)

    q = torch.rand(1, 1, 128, 128)
    q_lbl = (torch.rand(1, 1, 128, 128) > 0.5).float()
    s_img = torch.rand(1, 4, 1, 128, 128)
    s_msk = (torch.rand(1, 4, 1, 128, 128) > 0.5).float()

    opt.zero_grad()
    pred = m_proto(q, s_img, s_msk)
    loss = soft_dice_loss(pred, q_lbl)
    loss.backward()

    proto_grad_norm = m_proto.enc_blocks[1].cross.cross_conv.weight.grad[:, -2:, :, :].abs().sum().item()
    assert proto_grad_norm > 0.0, "Prototype channels received zero gradient!"
    opt.step()


if __name__ == "__main__":
    test_zero_init_equivalence_with_baseline()
    print("[PASS] test_zero_init_equivalence_with_baseline")
    test_permutation_invariance_over_s()
    print("[PASS] test_permutation_invariance_over_s")
    for s in [1, 4, 8, 64]:
        test_variable_support_sizes_and_empty_masks(s)
        print(f"[PASS] test_variable_support_sizes_and_empty_masks(S={s})")
    test_recipe_r_augmentations_binary_masks_and_range()
    print("[PASS] test_recipe_r_augmentations_binary_masks_and_range")
    test_one_training_step_proto_gradients()
    print("[PASS] test_one_training_step_proto_gradients")
    print("ALL R-B-U UNIT TESTS PASSED!")
