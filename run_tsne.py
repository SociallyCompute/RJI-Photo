import yaml
import argparse
import sys
import numpy as np
import pandas as pds
from tqdm import tqdm

from matplotlib import pyplot as plt

from models import *
from preprocessing import MissourianImageDataset

from scoring.latentspace import LatentSpaces
from scoring.reshape import Reshaper
np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='Generic runner for T-SNE Latent Space')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

parser.add_argument('--checkpoint', '-p',
                    dest="checkpoint",
                    metavar='FILE',
                    help='path to the checkpoint file',
                    default='logs/VanillaVAE/version_3/checkpoints/epoch=3-step=816.ckpt')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']].load_from_checkpoint(args.checkpoint,
                                                                        **config['model_params'])

ls = LatentSpaces(
    model=model,
    params=config['model_params']
)

data = MissourianImageDataset(
    **config["data_params"],
    pin_memory=len(config['trainer_params']['gpus']) != 0,
    include_cc=True
)
data.setup()

test_dataloader = data.test_dataloader()

paths = [str(p) for p in test_dataloader.dataset.imgs]

d = []
ys = []
for x, y in tqdm(test_dataloader):
    d.append((ls.get_latent_space(x, cc=y.detach().numpy())[0]).detach().numpy())
    ys.append(y.detach().numpy())

d = np.concatenate(d, axis=0)
y = np.concatenate(ys, axis=0)

ts = Reshaper()

transformed_data = ts.tsne(d, ret_data=True, n_components=3)[0]
print(transformed_data.shape)

df_dict = {
    "paths": paths,
    "dim1": transformed_data[:, 0],
    "dim2": transformed_data[:, 1],
    "dim3": transformed_data[:, 2],
    "code": y
}

df = pds.DataFrame(df_dict)

# TODO Fix this to be non-static and based on the model
df.to_csv("df3dimv4.csv")

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=y, cmap='Paired')
plt.colorbar()
plt.show()
