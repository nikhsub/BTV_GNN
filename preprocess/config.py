"""
Configuration file for preprocessing scripts.
Centralizes all parameters for better maintainability and reproducibility.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import yaml
import os

@dataclass
class TrackFeatureConfig:
    """Configuration for track features and their properties."""
    features: List[str]
    dummy_values: Dict[str, float]
    dtype_map: Dict[str, str]

@dataclass
class EdgeFilterConfig:
    """Configuration for edge filtering parameters."""
    # Training edge filters
    train_cptopv_max: float = 15.0
    train_dca_max: float = 0.125
    train_dca_sig_max: float = 30.0
    train_pvtoPCA_1_max: float = 15.0
    train_pvtoPCA_2_max: float = 15.0
    train_dotprod_1_min: float = 0.75
    train_dotprod_2_min: float = 0.75
    train_pair_invmass_max: float = 5.0
    train_pair_mom_max: float = 100.0
    
    # Validation edge filters
    val_cptopv_max: float = 50.0
    val_pvtoPCA_1_max: float = 50.0
    val_pvtoPCA_2_max: float = 50.0
    val_dotprod_1_min: float = 0.50
    val_dotprod_2_min: float = 0.50
    val_pair_invmass_max: float = 10.0
    
    # Test edge filters
    test_cptopv_max: float = 100.0
    test_pair_mom_max: float = 100.0

@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    batch_size: int = 100
    min_tracks: int = 2
    min_signal_tracks: int = 3
    default_hadron_weight: float = 1.0
    pickle_protocol: int = 4  # Use highest protocol for better compression
    
    # Memory management
    gc_frequency: int = 10  # Run garbage collection every N batches
    max_memory_gb: float = 8.0  # Maximum memory usage before forcing cleanup

@dataclass
class PreprocessConfig:
    """Main configuration class."""
    track_features: TrackFeatureConfig
    edge_filters: EdgeFilterConfig
    processing: ProcessingConfig
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> 'PreprocessConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PreprocessConfig':
        """Create configuration from dictionary."""
        track_config = TrackFeatureConfig(**config_dict.get('track_features', {}))
        edge_config = EdgeFilterConfig(**config_dict.get('edge_filters', {}))
        proc_config = ProcessingConfig(**config_dict.get('processing', {}))
        
        return cls(
            track_features=track_config,
            edge_filters=edge_config,
            processing=proc_config
        )
    
    def save_to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'track_features': {
                'features': self.track_features.features,
                'dummy_values': self.track_features.dummy_values,
                'dtype_map': self.track_features.dtype_map
            },
            'edge_filters': self.edge_filters.__dict__,
            'processing': self.processing.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

# Default configuration
DEFAULT_CONFIG = PreprocessConfig(
    track_features=TrackFeatureConfig(
        features=[
            'trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig',
            'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge'
        ],
        dummy_values={
            'trk_eta': -999.0, 'trk_phi': -999.0, 'trk_ip2d': -999.0, 'trk_ip3d': -999.0,
            'trk_ip2dsig': -999.0, 'trk_ip3dsig': -999.0, 'trk_p': -999.0, 'trk_pt': -999.0,
            'trk_nValid': -1, 'trk_nValidPixel': -1, 'trk_nValidStrip': -1, 'trk_charge': -3
        },
        dtype_map={
            'trk_eta': 'float32', 'trk_phi': 'float32', 'trk_ip2d': 'float32', 'trk_ip3d': 'float32',
            'trk_ip2dsig': 'float32', 'trk_ip3dsig': 'float32', 'trk_p': 'float32', 'trk_pt': 'float32',
            'trk_nValid': 'int16', 'trk_nValidPixel': 'int16', 'trk_nValidStrip': 'int16', 'trk_charge': 'int8'
        }
    ),
    edge_filters=EdgeFilterConfig(),
    processing=ProcessingConfig()
)

def get_config(config_path: str = None) -> PreprocessConfig:
    """Get configuration, loading from file if provided, otherwise use default."""
    if config_path and os.path.exists(config_path):
        return PreprocessConfig.load_from_yaml(config_path)
    return DEFAULT_CONFIG

# Edge feature names (consistent across all scripts)
EDGE_FEATURE_NAMES = [
    'dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1',
    'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass'
]

# Other feature names
OTHER_FEATURE_NAMES = [
    'sig_ind', 'sig_flag', 'bkg_flag', 'SVtrk_ind', 'had_pt',
    'trk_1', 'trk_2', 'deltaR', 'dca', 'dca_sig', 'cptopv',
    'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 
    'pair_mom', 'pair_invmass'
]