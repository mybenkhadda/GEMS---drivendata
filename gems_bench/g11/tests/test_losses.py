import pytest

torch = pytest.importorskip("torch")

from gems_bench.g11.training import losses as L  # noqa: E402


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 1, 16, 16, generator=g, requires_grad=True)
    y = (torch.rand(2, 1, 16, 16, generator=g) > 0.9).float()
    valid = torch.ones(2, 1, 16, 16)
    valid[:, :, :, :4] = 0
    return logits, y, valid


@pytest.mark.parametrize("fn", [L.weighted_bce_loss, L.focal_loss, L.soft_dice_loss, L.positive_unlabeled_loss])
def test_losses_finite_and_differentiable(fn):
    logits, y, valid = _batch()
    loss = fn(logits, y, valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (logits.grad[:, :, :, :4] == 0).all() or fn is L.soft_dice_loss   # masked pixels get no gradient


def test_masked_pixels_do_not_change_loss():
    logits, y, valid = _batch()
    a = L.weighted_bce_loss(logits, y, valid)
    l2 = logits.detach().clone()
    l2[:, :, :, :4] = 50.0
    assert torch.allclose(a, L.weighted_bce_loss(l2, y, valid))


def test_boundary_loss_and_pu():
    logits, y, valid = _batch()
    dist = torch.rand(2, 1, 16, 16) * 2000
    assert torch.isfinite(L.boundary_loss(logits, y, valid, dist))
    assert L.positive_unlabeled_loss(logits, y, valid) >= 0


def test_class_imbalance_pos_weight_increases_positive_loss():
    logits = torch.zeros(1, 1, 4, 4)
    y = torch.zeros(1, 1, 4, 4)
    y[0, 0, 0, 0] = 1
    v = torch.ones_like(y)
    assert L.weighted_bce_loss(logits, y, v, 10.0) > L.weighted_bce_loss(logits, y, v, 1.0)


@pytest.mark.parametrize("name", ["weighted_bce", "bce_dice", "focal", "bce_boundary"])
def test_loss_factory_and_multitask(name):
    logits, y, valid = _batch()
    batch = {"y": y, "valid": valid, "distance": torch.rand_like(y) * 1000,
             "orientation": torch.nn.functional.normalize(torch.randn(2, 2, 16, 16), dim=1)}
    out = {"logits": logits, "orientation": torch.tanh(torch.randn(2, 2, 16, 16)), "distance": torch.rand_like(y)}
    total, parts = L.masked_multitask_loss(out, batch, L.build_loss(name))
    assert torch.isfinite(total) and "main" in parts and "orientation" in parts and "distance" in parts
    with pytest.raises(ValueError):
        L.build_loss("nope")
