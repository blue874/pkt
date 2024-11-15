import hydra
import os
import sys
from omegaconf import DictConfig,OmegaConf
from src.utils import setup_ddp, cleanup_ddp, attachdebug, init_wandb, wandb_log, initialize_classification_head, set_seed
project_dir = os.getcwd()
sys.path.append(project_dir)
@hydra.main(config_path=os.path.join(project_dir, "config/DIST"), config_name="config.yaml")
def main(cfg:DictConfig):
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))
    cfg.model='model_student'
    print(OmegaConf.to_yaml(cfg))
