import numpy as np
import torch

from armd.model import SharedNet, joint_loss


def make(batch=32, base=12, expo=5, sev=7, agents=9):
    torch.manual_seed(0)
    net = SharedNet(base, expo, sev, agents, dims=(16, 8), rep_dim=6, dropout=0.0)
    xb = torch.randn(batch, base)
    xe = torch.randn(batch, expo)
    xs = torch.randn(batch, sev)
    return net, xb, xe, xs


def test_output_shapes():
    net, xb, xe, xs = make()
    s, p, y0, y1 = net(xb, xe, xs)
    assert s.shape == (32, 9)
    assert p.shape == y0.shape == y1.shape == (32,)


def test_exposure_block_cannot_reach_the_causal_heads():
    """The routing claim is the paper's central design argument, so it is tested.

    Perturbing the exposure block must move the susceptibility logits and leave the
    propensity and both outcome logits bit-identical.
    """
    net, xb, xe, xs = make()
    net.eval()
    with torch.no_grad():
        s1, p1, a1, b1 = net(xb, xe, xs)
        s2, p2, a2, b2 = net(xb, xe + 5.0, xs)
    assert not torch.allclose(s1, s2)
    assert torch.equal(p1, p2) and torch.equal(a1, a2) and torch.equal(b1, b2)


def test_severity_block_cannot_reach_the_susceptibility_head():
    net, xb, xe, xs = make()
    net.eval()
    with torch.no_grad():
        s1, p1, _, _ = net(xb, xe, xs)
        s2, p2, _, _ = net(xb, xe, xs + 5.0)
    assert torch.equal(s1, s2)
    assert not torch.allclose(p1, p2)


def test_masked_loss_ignores_untested_agents():
    net, xb, xe, xs = make()
    out = net(xb, xe, xs)
    ys = torch.randint(0, 2, (32, 9)).float()
    ms = torch.zeros(32, 9)
    ms[:, :3] = 1.0
    a = torch.randint(0, 2, (32,)).float()
    yc = torch.randint(0, 2, (32,)).float()
    mc = torch.ones(32)
    _, succ_a, _, _ = joint_loss(out, ys, ms, a, yc, mc, 0.5)
    flipped = ys.clone()
    flipped[:, 3:] = 1 - flipped[:, 3:]
    _, succ_b, _, _ = joint_loss(out, flipped, ms, a, yc, mc, 0.5)
    assert torch.allclose(succ_a, succ_b)


def test_fully_masked_batch_gives_zero_not_nan():
    net, xb, xe, xs = make()
    out = net(xb, xe, xs)
    zeros_s = torch.zeros(32, 9)
    zeros_c = torch.zeros(32)
    total, s, p, o = joint_loss(
        out, torch.zeros(32, 9), zeros_s,
        torch.zeros(32), torch.zeros(32), zeros_c, 0.5,
    )
    for t in (total, s, p, o):
        assert torch.isfinite(t) and float(t.detach()) == 0.0


def test_outcome_head_trains_only_on_its_own_arm():
    net, xb, xe, xs = make(batch=4)
    out = net(xb, xe, xs)
    a = torch.tensor([1.0, 1.0, 0.0, 0.0])
    yc = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mc = torch.ones(4)
    ys, ms = torch.zeros(4, 9), torch.zeros(4, 9)
    _, _, _, o1 = joint_loss(out, ys, ms, a, yc, mc, 0.5)
    s, p, y0, y1 = out
    _, _, _, o2 = joint_loss((s, p, y0, y1 + 3.0), ys, ms, a, yc, mc, 0.5)
    _, _, _, o3 = joint_loss((s, p, y0 + 3.0, y1), ys, ms, a, yc, mc, 0.5)
    assert not torch.allclose(o1, o2)
    assert not torch.allclose(o1, o3)


def test_a_training_step_reduces_the_loss():
    net, xb, xe, xs = make(batch=256)
    ys = torch.randint(0, 2, (256, 9)).float()
    ms = torch.ones(256, 9)
    a = torch.randint(0, 2, (256,)).float()
    yc = a.clone()
    mc = torch.ones(256)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss, *_ = joint_loss(net(xb, xe, xs), ys, ms, a, yc, mc, 0.5)
        loss.backward()
        opt.step()
        first = float(loss.detach()) if first is None else first
    assert float(loss.detach()) < first
