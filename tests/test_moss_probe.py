import torch
import torch.nn.functional as F
from mokrea.moss_probe import MossAttentionProbe, capture_moss_attention, moss_explicit_attention

def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin

def test_moss_explicit_attention_matches_sdpa():
    torch.manual_seed(19)
    b, length, heads, d = 2, 11, 4, 16
    q = torch.randn(b, length, heads*d)
    k, v = torch.randn_like(q), torch.randn_like(q)
    rope = torch.randn(1, length, d)
    qh = rotate(q.reshape(b,length,heads,d),rope).transpose(1,2)
    kh = rotate(k.reshape(b,length,heads,d),rope).transpose(1,2)
    vh = v.reshape(b,length,heads,d).transpose(1,2)
    expected = F.scaled_dot_product_attention(qh,kh,vh).transpose(1,2).reshape(b,length,heads*d)
    actual, maps = moss_explicit_attention(q,k,v,rope,heads)
    assert maps.shape == (b,heads,length,length)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

def test_moss_probe_restores_helper():
    class Dummy: pass
    module = Dummy()
    original = lambda q,k,v,rope,heads: q
    module._attention = original
    probe = MossAttentionProbe(mode="summary")
    q = torch.randn(1,5,16)
    with capture_moss_attention(module,probe):
        module._attention(q,q,q,torch.randn(1,5,4),4)
    assert module._attention is original
    assert probe.records[0].probabilities.shape == (1,4,1,5)
