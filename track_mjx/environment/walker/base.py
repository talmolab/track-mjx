"""
Defines high level abstracted walker class. All walker will inherit this high level class.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import jax.numpy as jp
from brax import math as brax_math
from brax.io import mjcf
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

class BaseWalker(ABC):
    """
    Abstract base class for different types of walker models (rodents, flies, mice, etc.).
    Defines the interface that all walker implementations must follow.
    """
    
    def __init__(self, xml_path: str, joint_names: List[str], 
                 body_names: List[str], end_eff_names: List[str],
                 torque_actuators: bool = False, rescale_factor: float = 1.0):
        """
        Initialize the walker with model configuration.
        
        Args:
            xml_path: Path to the MJCF XML model file
            joint_names: List of joint names in the model
            body_names: List of body part names
            end_eff_names: List of end effector names
            torque_actuators: Whether to use torque actuators
            rescale_factor: Factor to rescale the model
        """
        self._xml_path = xml_path
        self._joint_names = joint_names
        self._body_names = body_names
        self._end_eff_names = end_eff_names
        
        # Load and configure the model
        self._mjcf_model = self._load_mjcf_model(torque_actuators, rescale_factor)
        self.sys = mjcf.load_model(self._mjcf_model.model.ptr)
        self._initialize_indices()
    
    @abstractmethod
    def _load_mjcf_model(self, torque_actuators: bool, rescale_factor: float) -> mjcf_dm.Physics:
        """
        Load and configure the MJCF model with optional torque actuators and rescaling.
        
        Args:
            torque_actuators: Whether to use torque actuators
            rescale_factor: Factor to rescale the model
            
        Returns:
            Configured MuJoCo physics object
        """
        pass
    
    @abstractmethod
    def _initialize_indices(self) -> None:
        """
        Initialize indices for joints, bodies, and end-effectors.
        Each implementation should set at minimum:
        - self._joint_idxs
        - self._body_idxs
        - self._endeff_idxs
        - self._torso_idx
        """
        pass
    
    @property
    def joint_idxs(self) -> jp.ndarray:
        """Get joint indices."""
        return self._joint_idxs
    
    @property
    def body_idxs(self) -> jp.ndarray:
        """Get body indices."""
        return self._body_idxs
    
    @property
    def endeff_idxs(self) -> jp.ndarray:
        """Get end effector indices."""
        return self._endeff_idxs
    
    @property
    def torso_idx(self) -> jp.ndarray:
        """Get torso index."""
        return self._torso_idx
        
    def get_joint_positions(self, qpos: jp.ndarray) -> jp.ndarray:
        """
        Get joint positions from state.
        
        Args:
            qpos: Full position state vector
            
        Returns:
            Joint positions vector
        """
        return qpos[self.joint_idxs]
    
    def get_body_positions(self, xpos: jp.ndarray) -> jp.ndarray:
        """
        Get body positions from state.
        
        Args:
            xpos: Full position state vector
            
        Returns:
            Body positions vector
        """
        return xpos[self.body_idxs]
    
    def get_end_effector_positions(self, xpos: jp.ndarray) -> jp.ndarray:
        """
        Get end effector positions from state.
        
        Args:
            xpos: Full position state vector
            
        Returns:
            End effector positions vector
        """
        return xpos[self.endeff_idxs]
    
    def get_torso_position(self, xpos: jp.ndarray) -> jp.ndarray:
        """
        Get torso position from state.
        
        Args:
            xpos: Full position state vector
            
        Returns:
            Torso position vector
        """
        return xpos[self.torso_idx]
    
    @abstractmethod
    def default_joint_angles(self) -> jp.ndarray:
        """
        Get default joint angles for initialization.
        
        Returns:
            Default joint angles vector
        """
        pass
    
    @abstractmethod
    def joint_limits(self) -> Tuple[jp.ndarray, jp.ndarray]:
        """
        Get joint angle limits.
        
        Returns:
            Tuple of (lower_limits, upper_limits)
        """
        pass