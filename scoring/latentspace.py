from pathlib import Path

from models import BaseVAE
from typing import TypeVar, Union

Tensor = TypeVar("torch.tensor")


class LatentSpaces(object):
    def __init__(
            self,
            model: BaseVAE,
            params: dict,
            **kwargs
    ):
        self.model = model
        self.params = params

    def get_latent_space(self, x: Tensor, **kwargs) -> Tensor:
        colorcode = kwargs['cc']
        q, p, mu, logvar = self.model(x, **kwargs)
        return [mu, logvar, colorcode]

    def load_model(self, checkpoint_file: Union[str, Path]) -> None:
        self.model.load_from_checkpoint(checkpoint_file, params=self.params)
