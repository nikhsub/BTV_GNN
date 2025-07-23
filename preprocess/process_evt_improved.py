import warnings
warnings.filterwarnings("ignore")
import argparse
import uproot
import numpy as np
import torch
import pickle
from torch_geometric.data import Data
import gc
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("Creating event-level training samples")
parser.add_argument("-d", "--data", required=True, help="Data file")
parser.add_argument("-st", "--save_tag", required=True, help="Save tag for data")
parser.add_argument("-s", "--start", type=int, default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", type=int, default=4000, help="Evt # to end with")
parser.add_argument("-b", "--batch_size", type=int, default=100, help="Batch size for processing")
parser.add_argument("--min_tracks", type=int, default=2, help="Minimum tracks required for event")
parser.add_argument("--output_dir", default=".", help="Output directory for pickle files")
args = parser.parse_args()

# Track features with type hints
TRK_FEATURES = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 
                'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

# Feature-specific dummy values for better handling of missing data
DUMMY_VALUES = {
    'trk_eta': -999.0, 'trk_phi': -999.0, 'trk_ip2d': -999.0, 'trk_ip3d': -999.0,
    'trk_ip2dsig': -999.0, 'trk_ip3dsig': -999.0, 'trk_p': -999.0, 'trk_pt': -999.0,
    'trk_nValid': -1, 'trk_nValidPixel': -1, 'trk_nValidStrip': -1, 'trk_charge': -3
}

class EventProcessor:
    """Enhanced event processor with better memory management and error handling."""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.datatree = None
        self.num_evts = 0
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data with error handling."""
        try:
            logger.info(f"Loading data from {self.data_file}")
            self.file_handle = uproot.open(self.data_file)
            self.datatree = self.file_handle['tree']
            self.num_evts = self.datatree.num_entries
            logger.info(f"Successfully loaded {self.num_evts} events")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def load_event_batch(self, start: int, end: int) -> Dict:
        """Load a batch of events to reduce memory usage."""
        try:
            batch_data = {}
            
            # Load track features
            for feat in TRK_FEATURES:
                batch_data[feat] = self.datatree[feat].array(entry_start=start, entry_stop=end)
            
            # Load other arrays
            other_features = ['sig_ind', 'sig_flag', 'bkg_flag', 'SVtrk_ind', 'had_pt',
                            'trk_1', 'trk_2', 'deltaR', 'dca', 'dca_sig', 'cptopv',
                            'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 
                            'pair_mom', 'pair_invmass']
            
            for feat in other_features:
                batch_data[feat] = self.datatree[feat].array(entry_start=start, entry_stop=end)
            
            return batch_data
        except Exception as e:
            logger.error(f"Failed to load batch {start}-{end}: {e}")
            raise
    
    def create_edge_index(self, trk_1, trk_2, dca, deltaR, dca_sig, cptopv, 
                         pvtoPCA_1, pvtoPCA_2, dotprod_1, dotprod_2, 
                         pair_mom, pair_invmass, val_comb_inds) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized edge creation with better vectorization and parameter validation.
        """
        # Convert to numpy arrays efficiently
        arrays = {
            'trk_1': trk_1.to_numpy() if hasattr(trk_1, 'to_numpy') else np.array(trk_1),
            'trk_2': trk_2.to_numpy() if hasattr(trk_2, 'to_numpy') else np.array(trk_2),
            'dca': dca.to_numpy() if hasattr(dca, 'to_numpy') else np.array(dca),
            'deltaR': deltaR.to_numpy() if hasattr(deltaR, 'to_numpy') else np.array(deltaR),
            'dca_sig': dca_sig.to_numpy() if hasattr(dca_sig, 'to_numpy') else np.array(dca_sig),
            'cptopv': cptopv.to_numpy() if hasattr(cptopv, 'to_numpy') else np.array(cptopv),
            'pvtoPCA_1': pvtoPCA_1.to_numpy() if hasattr(pvtoPCA_1, 'to_numpy') else np.array(pvtoPCA_1),
            'pvtoPCA_2': pvtoPCA_2.to_numpy() if hasattr(pvtoPCA_2, 'to_numpy') else np.array(pvtoPCA_2),
            'dotprod_1': dotprod_1.to_numpy() if hasattr(dotprod_1, 'to_numpy') else np.array(dotprod_1),
            'dotprod_2': dotprod_2.to_numpy() if hasattr(dotprod_2, 'to_numpy') else np.array(dotprod_2),
            'pair_mom': pair_mom.to_numpy() if hasattr(pair_mom, 'to_numpy') else np.array(pair_mom),
            'pair_invmass': pair_invmass.to_numpy() if hasattr(pair_invmass, 'to_numpy') else np.array(pair_invmass)
        }
        
        # Validate track indices
        valid_edge_mask = (
            np.isin(arrays['trk_1'], val_comb_inds) & 
            np.isin(arrays['trk_2'], val_comb_inds)
        )
        
        # Improved feature selection with configurable thresholds
        feature_mask = (
            (arrays['cptopv'] < 15) &
            (arrays['dca'] < 0.125) &
            (arrays['dca_sig'] < 30) &
            (arrays['pvtoPCA_1'] < 15) &
            (arrays['pvtoPCA_2'] < 15) &
            (np.abs(arrays['dotprod_1']) > 0.75) &
            (np.abs(arrays['dotprod_2']) > 0.75) &
            (arrays['pair_invmass'] < 5) &
            (arrays['pair_mom'] < 100) &
            np.isfinite(arrays['dca']) &  # Check for valid values
            np.isfinite(arrays['deltaR'])
        )
        
        final_mask = valid_edge_mask & feature_mask
        
        if not np.any(final_mask):
            return np.array([[], []], dtype=np.int64), np.array([]).reshape(0, 10)
        
        # Extract edges and features
        edge_index = np.vstack([
            arrays['trk_1'][final_mask], 
            arrays['trk_2'][final_mask]
        ]).astype(np.int64)
        
        edge_features = np.column_stack([
            arrays['dca'][final_mask], arrays['deltaR'][final_mask],
            arrays['dca_sig'][final_mask], arrays['cptopv'][final_mask],
            arrays['pvtoPCA_1'][final_mask], arrays['pvtoPCA_2'][final_mask], 
            arrays['dotprod_1'][final_mask], arrays['dotprod_2'][final_mask], 
            arrays['pair_mom'][final_mask], arrays['pair_invmass'][final_mask]
        ])
        
        return edge_index, edge_features
    
    def process_event(self, evt_idx: int, batch_data: Dict) -> Optional[Data]:
        """Process a single event with improved error handling."""
        try:
            # Extract event features
            evt_features = {f: batch_data[f][evt_idx] for f in TRK_FEATURES}
            
            # Create feature matrix with proper dtype handling
            feature_list = []
            for f in TRK_FEATURES:
                feat_data = evt_features[f]
                if hasattr(feat_data, 'to_numpy'):
                    feat_array = feat_data.to_numpy()
                else:
                    feat_array = np.array(feat_data)
                feature_list.append(feat_array)
            
            fullfeatmat = np.column_stack(feature_list).astype(np.float32)
            
            # Handle missing values with feature-specific dummy values
            dummy_array = np.array([DUMMY_VALUES[f] for f in TRK_FEATURES], dtype=np.float32)
            mask = ~np.isfinite(fullfeatmat)
            fullfeatmat[mask] = np.broadcast_to(dummy_array, fullfeatmat.shape)[mask]
            
            # Get valid indices
            valid_indices = np.arange(len(fullfeatmat))
            val_inds_map = {ind: ind for ind in valid_indices}
            
            # Get signal indices
            evtsiginds = list(set(batch_data['sig_ind'][evt_idx]))
            evtsiginds = [val_inds_map[ind] for ind in evtsiginds if ind in val_inds_map]
            
            # Skip events with insufficient signal tracks
            if len(evtsiginds) < args.min_tracks:
                return None
            
            # Create edges
            edge_index, edge_features = self.create_edge_index(
                batch_data['trk_1'][evt_idx], batch_data['trk_2'][evt_idx],
                batch_data['dca'][evt_idx], batch_data['deltaR'][evt_idx],
                batch_data['dca_sig'][evt_idx], batch_data['cptopv'][evt_idx],
                batch_data['pvtoPCA_1'][evt_idx], batch_data['pvtoPCA_2'][evt_idx],
                batch_data['dotprod_1'][evt_idx], batch_data['dotprod_2'][evt_idx],
                batch_data['pair_mom'][evt_idx], batch_data['pair_invmass'][evt_idx],
                valid_indices
            )
            
            if edge_index.shape[1] == 0:
                return None
            
            # Remap edge indices
            edge_index = np.vectorize(val_inds_map.get)(edge_index)
            
            # Create labels
            labels = np.zeros(len(fullfeatmat), dtype=np.float32)
            labels[evtsiginds] = 1
            
            # Create graph data
            evt_graph = Data(
                x=torch.tensor(fullfeatmat, dtype=torch.float),
                y=torch.tensor(labels, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.int64),
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                had_weight=torch.tensor([1.0], dtype=torch.float),
                event_id=evt_idx  # Add event ID for tracking
            )
            
            return evt_graph
            
        except Exception as e:
            logger.warning(f"Failed to process event {evt_idx}: {e}")
            return None
    
    def process_events_in_batches(self) -> List[Data]:
        """Process events in batches for better memory management."""
        event_graphs = []
        
        start = args.start
        end = args.end if args.end != -1 else self.num_evts
        
        logger.info(f"Processing events {start} to {end}")
        
        for batch_start in range(start, end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, end)
            
            try:
                # Load batch data
                batch_data = self.load_event_batch(batch_start, batch_end)
                
                # Process each event in the batch
                for i in range(batch_end - batch_start):
                    evt_graph = self.process_event(i, batch_data)
                    if evt_graph is not None:
                        event_graphs.append(evt_graph)
                
                # Clean up memory
                del batch_data
                gc.collect()
                
                logger.info(f"Processed batch {batch_start}-{batch_end}, "
                          f"total graphs: {len(event_graphs)}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_start}-{batch_end}: {e}")
                continue
        
        return event_graphs
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'file_handle'):
            self.file_handle.close()

def main():
    """Main processing function with improved error handling."""
    processor = None
    try:
        # Create processor
        processor = EventProcessor(args.data)
        
        # Process events
        logger.info("Starting event processing...")
        event_graphs = processor.process_events_in_batches()
        
        # Save results
        output_file = f"{args.output_dir}/evttraindata_{args.save_tag}.pkl"
        logger.info(f"Saving {len(event_graphs)} event graphs to {output_file}")
        
        with open(output_file, 'wb') as f:
            pickle.dump(event_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    finally:
        if processor:
            processor.close()

if __name__ == "__main__":
    main()