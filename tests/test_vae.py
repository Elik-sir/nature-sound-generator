import torch

from src.models.vae import VAESynthesizer


def test_vae_forward_shapes():
    model = VAESynthesizer()
    x = torch.randn(2, 1, 128, 128)
    recon, mu, logvar = model(x)
    assert recon.shape == (2, 1, 128, 128)
    assert mu.shape == (2, model.latent_dim)
    assert logvar.shape == (2, model.latent_dim)


def test_vae_reparameterize_shape():
    model = VAESynthesizer(latent_dim=32)
    mu = torch.zeros(4, 32)
    logvar = torch.zeros(4, 32)
    z = model.reparameterize(mu, logvar)
    assert z.shape == (4, 32)
