import os
import yaml
import argparse
from pathlib import Path
from models import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from preprocessing import MissourianImageDataset

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'], params=config['exp_params'])

data = MissourianImageDataset(**config["data_params"], pin_memory=len(config['trainer_params']['accelerator']) != 'cpu')

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(logging_interval='epoch', log_momentum=True),
                     ModelCheckpoint(save_top_k=2,
                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                     monitor="val_loss",
                                     save_last=True),
                 ],
                 strategy='ddp',
                 **config['trainer_params'])

Path("{}/Samples".format(tb_logger.log_dir)).mkdir(exist_ok=True, parents=True)
Path("{}/Reconstructions".format(tb_logger.log_dir)).mkdir(exist_ok=True, parents=True)

print("======= Tuning {} =======".format(config['model_params']['name']))
runner.tune(model=model, datamodule=data)

print("======= Training {} =======".format(config['model_params']['name']))
runner.fit(model=model, datamodule=data)
