"""
Data validation and quality checking utilities for preprocessing scripts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging
from pathlib import Path
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation and quality assessment."""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.validation_results = {}
    
    def validate_feature_matrix(self, feature_matrix: np.ndarray, 
                               feature_names: List[str]) -> Dict[str, Any]:
        """Validate feature matrix for common issues."""
        results = {}
        
        # Basic statistics
        results['shape'] = feature_matrix.shape
        results['dtype'] = feature_matrix.dtype
        results['memory_usage_mb'] = feature_matrix.nbytes / 1024 / 1024
        
        # Missing values
        nan_mask = np.isnan(feature_matrix)
        inf_mask = np.isinf(feature_matrix)
        
        results['nan_stats'] = {
            'total_nans': np.sum(nan_mask),
            'nan_percentage': np.sum(nan_mask) / feature_matrix.size * 100,
            'features_with_nans': [feature_names[i] for i in range(len(feature_names)) 
                                  if np.any(nan_mask[:, i])]
        }
        
        results['inf_stats'] = {
            'total_infs': np.sum(inf_mask),
            'inf_percentage': np.sum(inf_mask) / feature_matrix.size * 100,
            'features_with_infs': [feature_names[i] for i in range(len(feature_names)) 
                                  if np.any(inf_mask[:, i])]
        }
        
        # Feature-wise statistics
        feature_stats = {}
        for i, name in enumerate(feature_names):
            col_data = feature_matrix[:, i]
            valid_data = col_data[np.isfinite(col_data)]
            
            if len(valid_data) > 0:
                feature_stats[name] = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'median': float(np.median(valid_data)),
                    'valid_count': len(valid_data),
                    'invalid_count': len(col_data) - len(valid_data)
                }
            else:
                feature_stats[name] = {
                    'mean': None, 'std': None, 'min': None, 'max': None,
                    'median': None, 'valid_count': 0, 'invalid_count': len(col_data)
                }
        
        results['feature_stats'] = feature_stats
        
        return results
    
    def validate_edge_features(self, edge_features: np.ndarray) -> Dict[str, Any]:
        """Validate edge features."""
        results = {}
        
        if edge_features.size == 0:
            results['empty'] = True
            return results
        
        results['empty'] = False
        results['shape'] = edge_features.shape
        results['dtype'] = edge_features.dtype
        
        # Check for invalid values
        nan_count = np.sum(np.isnan(edge_features))
        inf_count = np.sum(np.isinf(edge_features))
        
        results['nan_count'] = int(nan_count)
        results['inf_count'] = int(inf_count)
        results['valid_percentage'] = float((edge_features.size - nan_count - inf_count) / edge_features.size * 100)
        
        # Statistical summary
        valid_data = edge_features[np.isfinite(edge_features)]
        if len(valid_data) > 0:
            results['stats'] = {
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data))
            }
        
        return results
    
    def validate_graph_data(self, graph_data: List[Data]) -> Dict[str, Any]:
        """Validate processed graph data."""
        results = {}
        
        if not graph_data:
            results['empty'] = True
            return results
        
        results['empty'] = False
        results['num_graphs'] = len(graph_data)
        
        # Collect statistics
        node_counts = []
        edge_counts = []
        feature_dims = []
        edge_feature_dims = []
        label_stats = []
        
        for graph in graph_data:
            node_counts.append(graph.x.shape[0])
            edge_counts.append(graph.edge_index.shape[1])
            feature_dims.append(graph.x.shape[1])
            
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                edge_feature_dims.append(graph.edge_attr.shape[1])
            
            if hasattr(graph, 'y') and graph.y is not None:
                labels = graph.y.numpy() if isinstance(graph.y, torch.Tensor) else graph.y
                label_stats.append({
                    'positive_ratio': float(np.mean(labels)),
                    'total_labels': len(labels)
                })
        
        # Summary statistics
        results['node_stats'] = {
            'mean': float(np.mean(node_counts)),
            'std': float(np.std(node_counts)),
            'min': int(np.min(node_counts)),
            'max': int(np.max(node_counts)),
            'median': float(np.median(node_counts))
        }
        
        results['edge_stats'] = {
            'mean': float(np.mean(edge_counts)),
            'std': float(np.std(edge_counts)),
            'min': int(np.min(edge_counts)),
            'max': int(np.max(edge_counts)),
            'median': float(np.median(edge_counts))
        }
        
        if feature_dims:
            results['feature_dimension'] = feature_dims[0]
            results['consistent_feature_dims'] = len(set(feature_dims)) == 1
        
        if edge_feature_dims:
            results['edge_feature_dimension'] = edge_feature_dims[0]
            results['consistent_edge_feature_dims'] = len(set(edge_feature_dims)) == 1
        
        if label_stats:
            pos_ratios = [s['positive_ratio'] for s in label_stats]
            results['label_stats'] = {
                'mean_positive_ratio': float(np.mean(pos_ratios)),
                'std_positive_ratio': float(np.std(pos_ratios)),
                'min_positive_ratio': float(np.min(pos_ratios)),
                'max_positive_ratio': float(np.max(pos_ratios))
            }
        
        return results
    
    def detect_outliers(self, data: np.ndarray, method: str = 'iqr', 
                       threshold: float = 1.5) -> np.ndarray:
        """Detect outliers in data using specified method."""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return z_scores > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def generate_feature_plots(self, feature_matrix: np.ndarray, 
                              feature_names: List[str], 
                              save_prefix: str = "features"):
        """Generate diagnostic plots for features."""
        n_features = len(feature_names)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Distribution plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, (name, ax) in enumerate(zip(feature_names, axes)):
            if i < n_features:
                data = feature_matrix[:, i]
                valid_data = data[np.isfinite(data)]
                
                if len(valid_data) > 0:
                    ax.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{name}\n(valid: {len(valid_data)}/{len(data)})')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                else:
                    ax.text(0.5, 0.5, 'No valid data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{name}\n(no valid data)')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Correlation matrix
        valid_mask = np.all(np.isfinite(feature_matrix), axis=1)
        if np.sum(valid_mask) > 1:
            valid_features = feature_matrix[valid_mask]
            correlation_matrix = np.corrcoef(valid_features.T)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, 
                       xticklabels=feature_names, 
                       yticklabels=feature_names,
                       annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_correlation.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def validate_preprocessing_results(self, pickle_file: str) -> Dict[str, Any]:
        """Comprehensive validation of preprocessing results."""
        logger.info(f"Validating preprocessing results from {pickle_file}")
        
        # Load data
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            return {'error': f"Failed to load pickle file: {e}"}
        
        # Validate based on data type
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], Data):
                # Graph data
                results = self.validate_graph_data(data)
                results['data_type'] = 'graph_data'
                
                # Additional graph-specific checks
                if not results.get('empty', True):
                    sample_graph = data[0]
                    if hasattr(sample_graph, 'x'):
                        feature_names = [f"feature_{i}" for i in range(sample_graph.x.shape[1])]
                        feature_results = self.validate_feature_matrix(
                            sample_graph.x.numpy(), feature_names
                        )
                        results['sample_features'] = feature_results
            else:
                results = {'data_type': 'unknown_list', 'length': len(data)}
        else:
            results = {'data_type': 'unknown', 'empty': True}
        
        # Save results
        results_file = self.output_dir / f"validation_{Path(pickle_file).stem}.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation results saved to {results_file}")
        return results
    
    def compare_preprocessing_runs(self, file_list: List[str]) -> Dict[str, Any]:
        """Compare multiple preprocessing runs."""
        comparison_results = {}
        
        for file_path in file_list:
            file_key = Path(file_path).stem
            try:
                results = self.validate_preprocessing_results(file_path)
                comparison_results[file_key] = results
            except Exception as e:
                comparison_results[file_key] = {'error': str(e)}
        
        # Generate comparison summary
        summary = {
            'num_files': len(file_list),
            'successful_validations': sum(1 for r in comparison_results.values() 
                                        if 'error' not in r),
            'failed_validations': sum(1 for r in comparison_results.values() 
                                    if 'error' in r)
        }
        
        comparison_results['summary'] = summary
        
        # Save comparison results
        comparison_file = self.output_dir / "preprocessing_comparison.json"
        import json
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        return comparison_results

def main():
    """Main function for standalone validation."""
    import argparse
    
    parser = argparse.ArgumentParser("Data validation for preprocessing results")
    parser.add_argument("-f", "--files", nargs="+", required=True, 
                       help="Pickle files to validate")
    parser.add_argument("-o", "--output_dir", default="validation_results",
                       help="Output directory for validation results")
    args = parser.parse_args()
    
    validator = DataValidator(args.output_dir)
    
    if len(args.files) == 1:
        results = validator.validate_preprocessing_results(args.files[0])
        print(f"Validation completed. Results: {results}")
    else:
        results = validator.compare_preprocessing_runs(args.files)
        print(f"Comparison completed. Summary: {results['summary']}")

if __name__ == "__main__":
    main()