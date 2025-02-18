import torch
from data.datamodule import *
from model.module import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import argparse
from omegaconf import OmegaConf
import time
import os
from pytorch_lightning.strategies.ddp import DDPStrategy


curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(cfg):
    pl.seed_everything(cfg.seed)
    if cfg.debug:
        wandbmode = 'offline'
        cfg.trainer.max_epochs = 1
    else:
        wandbmode = 'online'

    #############################
    # Set loggers and callbacks #
    #############################
    # Set loggers
    loggers = []
    loggers.append(WandbLogger(name=cfg.exp_name, project='ion-con', mode=wandbmode))
    loggers.append(CSVLogger(os.path.join(cfg.output_dir, cfg.exp_name, 'logs/'), name=cfg.exp_name))

    # Set callbacks
    callbacks = []
    callbacks.append(EarlyStopping(monitor=cfg.callback.early_stopping.monitor, patience=cfg.callback.early_stopping.patience))
    callbacks.append(ModelCheckpoint(
        dirpath=os.path.join(cfg.output_dir, cfg.exp_name, 'checkpoints/'),
        filename=cfg.exp_name+'-{epoch:02d}-{val_loss:.5f}',
        monitor=cfg.callback.model_checkpoint.monitor,
        save_top_k=cfg.callback.model_checkpoint.save_top_k,
        mode=cfg.callback.model_checkpoint.mode
    ))
    callbacks.append(LearningRateMonitor(logging_interval=cfg.callback.learning_rate_monitor.logging_interval))

    ####################
    # Load data module #
    ####################
    dm = IonConductivityDataModule(cfg.data)
    dm.setup()
    
    ################
    # Define model #
    ################
    model_classes = {
        'CGCNN': IonConductivityCGCNNModule,
        'PaiNN': PaiNN,
        'MolCLR': MolCLR,
        'MACE': MACE,
    }
    model_type = model_classes.get(cfg.model.name)
    
    # If you want to resume training, load the checkpoint. First, find cfg have attribute resume_from_checkpoint
    if hasattr(cfg, 'resume_from_checkpoint'):
        model = model_type.load_from_checkpoint(cfg.resume_from_checkpoint, cfg=cfg, map_location='cpu')
    else:
        model = model_type(cfg)

    ################
    # Train model  #
    ################
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,#DDPStrategy(find_unused_parameters=True),# cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs, 
        precision=cfg.trainer.precision,
        logger=loggers,
        callbacks=callbacks,
        # val_check_interval=0.1,
        # gradient_clip_val=cfg.trainer.gradient_clip_val,
    )
    
    # train
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # test
    test_loader = dm.test_dataloader()
    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    # Setup configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=os.path.join(curr_dir, 'config', 'config.yaml'))
    parser.add_argument('--output_path', type=str, default=os.path.join(curr_dir, 'output'))

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load(args.config_path)
    cfg.output_dir = args.output_path
    
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))
    main(cfg)