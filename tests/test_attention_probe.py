import torch
import torch.nn.functional as F

from mokrea.attention_probe import (
    KreaAttentionProbe,
    capture_krea_attention,
    explicit_attention,
)


def test_explicit_attention_matches_sdpa_without_gqa():
    torch.manual_seed(7)
    q = torch.randn(2, 4, 13, 16)
    k = torch.randn(2, 4, 13, 16)
    v = torch.randn(2, 4, 13, 16)
    expected = F.scaled_dot_product_attention(q, k, v)
    actual, probabilities = explicit_attention(q, k, v)
    assert probabilities.shape == (2, 4, 13, 13)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_explicit_attention_matches_sdpa_gqa():
    torch.manual_seed(11)
    q = torch.randn(2, 8, 9, 32)
    k = torch.randn(2, 2, 9, 32)
    v = torch.randn(2, 2, 9, 32)
    expected = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    actual, _ = explicit_attention(q, k, v, gqa=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_boolean_mask_matches_sdpa():
    torch.manual_seed(13)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    mask[..., -1] = False
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    actual, _ = explicit_attention(q, k, v, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_capture_restores_original_function():
    class Dummy:
        pass

    module = Dummy()
    original = lambda q, k, v, mask=None, scale=None, gqa=False: q
    module.attention = original
    probe = KreaAttentionProbe(mode="summary")
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    with capture_krea_attention(module, probe):
        module.attention(q, k, v, gqa=True)
    assert module.attention is original
    assert len(probe.records) == 1
    assert probe.records[0].probabilities.shape == (1, 2, 1, 4)
