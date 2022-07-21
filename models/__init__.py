from .basevae import BaseVAE
from .vae import VanillaVAE
from .beta_vae import BetaVAE
from .baseconvnet import BaseConvNet


vae_models = {
    'VanillaVAE': VanillaVAE,
    'BetaVAE': BetaVAE
}
