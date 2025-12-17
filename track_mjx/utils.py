from typing import Any
from omegaconf import DictConfig, OmegaConf
from vnl_playground.tasks.rodent import consts as rodent_consts
from vnl_playground.tasks.fruitfly import consts as fruitfly_consts
from vnl_playground.tasks.mouse import consts as mouse_consts
from ml_collections import config_dict
import logging

def prepare_config(cfg: DictConfig) -> tuple[DictConfig, Any, config_dict.ConfigDict]:

    # Determine walker
    walker_name = cfg.walker_config.walker_name

    # TODO: Add other walkers
    try:
        if walker_name == "rodent":
            logging.info("Using rodent walker")
            walker_xml_path = str(rodent_consts.RODENT_XML_PATH)
            arena_xml_path = str(rodent_consts.ARENA_XML_PATH)
            reference_data_path = str(rodent_consts.IMITATION_REFERENCE_PATH)
        elif walker_name == "fruitfly":
            logging.info("Using fruitfly walker")
            walker_xml_path = str(fruitfly_consts.FRUITFLY_XML_PATH)
            arena_xml_path = str(fruitfly_consts.ARENA_XML_PATH)
            raise NotImplementedError("Fruitfly reference data path not implemented yet.")
        elif walker_name == "celegans":
            logging.info("Using celegans walker")
            # TODO: Add celegans constants
            raise NotImplementedError("Celegans walker not implemented yet.")
        elif walker_name =="mouse":
            logging.info("Using mouse walker")
            walker_xml_path = str(mouse_consts.MOUSE_XML_PATH)
            raise NotImplementedError("Mouse arena and reference data paths not implemented yet.")
        elif walker_name == "stickbug":
            logging.info("Using stickbug walker")
            raise NotImplementedError("Stickbug walker not implemented yet.")
        else:
            raise ValueError(f"Unknown walker name: {walker_name}")
    except Exception as e:
        logging.error(f"Error determining walker constants: {e}")
        raise e

    # Update env_config with paths and walker settings
    OmegaConf.set_struct(cfg.env_config, False)
    OmegaConf.update(cfg.env_config, "walker_xml_path", walker_xml_path, merge=False)
    OmegaConf.update(cfg.env_config, "arena_xml_path", arena_xml_path, merge=False)
    OmegaConf.update(cfg.env_config, "reference_data_path", reference_data_path, merge=False)
    OmegaConf.update(cfg.env_config, "walker_name", cfg.walker_config.walker_name, merge=False)
    OmegaConf.update(cfg.env_config, "torque_actuators", cfg.walker_config.torque_actuators, merge=False)
    OmegaConf.update(cfg.env_config, "rescale_factor", cfg.walker_config.rescale_factor, merge=False)
    OmegaConf.set_struct(cfg.env_config, True)

    # Create ml_collections env_config
    env_cfg_dict = OmegaConf.to_container(cfg.env_config, resolve=True)
    env_cfg_ml = config_dict.ConfigDict(env_cfg_dict)

    # Convert config to dict
    logging.info(f"Configs: {OmegaConf.to_container(cfg, resolve=True)}")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    return (cfg, cfg_dict, env_cfg_ml)

def update_config(cfg: DictConfig, overrides: dict) -> DictConfig:
    
    OmegaConf.set_struct(cfg, False)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    OmegaConf.set_struct(cfg, True)
    return cfg