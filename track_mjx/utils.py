from omegaconf import DictConfig, OmegaConf
from vnl_mjx.tasks.rodent import consts as rodent_consts
from ml_collections import config_dict
import logging

def prepare_config(cfg: DictConfig) -> tuple[DictConfig, DictConfig, DictConfig, config_dict.ConfigDict, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:

    # Update env_config with paths and walker settings
    OmegaConf.set_struct(cfg.env_config, False)
    OmegaConf.update(cfg.env_config, "walker_xml_path", str(rodent_consts.RODENT_XML_PATH), merge=False)
    OmegaConf.update(cfg.env_config, "arena_xml_path", str(rodent_consts.ARENA_XML_PATH), merge=False)
    OmegaConf.update(cfg.env_config, "reference_data_path", str(rodent_consts.IMITATION_REFERENCE_PATH), merge=False)
    OmegaConf.update(cfg.env_config, "walker_name", cfg.walker_config.walker_name, merge=False)
    OmegaConf.update(cfg.env_config, "torque_actuators", cfg.walker_config.torque_actuators, merge=False)
    OmegaConf.update(cfg.env_config, "rescale_factor", cfg.walker_config.rescale_factor, merge=False)
    OmegaConf.set_struct(cfg.env_config, True)

    # Breakup config
    env_cfg = cfg.env_config
    render_cfg = cfg.render_config
    network_cfg = cfg.network_config
    train_setup = cfg.train_setup
    train_cfg = cfg.train_setup.train_config
    logging_cfg = cfg.logging_config
    walker_cfg = cfg.walker_config

    # Create ml_collections env_config
    env_cfg_dict = OmegaConf.to_container(env_cfg, resolve=True)
    env_cfg_ml = config_dict.ConfigDict(env_cfg_dict)

    # Convert config to dict
    logging.info(f"Configs: {OmegaConf.to_container(cfg, resolve=True)}")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    return (cfg, cfg_dict, env_cfg, env_cfg_ml, render_cfg, network_cfg, train_setup, train_cfg, logging_cfg, walker_cfg)

def update_config(cfg: DictConfig, overrides: dict) -> DictConfig:
    
    OmegaConf.set_struct(cfg, False)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    OmegaConf.set_struct(cfg, True)
    return cfg