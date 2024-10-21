import argparse
import uproot
import numpy as np
import torch
import pickle
import torch_geometric
from torch_geometric.data import Data, DataLoader
import random

parser = argparse.ArgumentParser("Creating labels and seeds")

parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-l", "--label", default="", help="Label file")
parser.add_argument("-st", "--save_tag", default="", help="Save tag for data")
#parser.add_argument("-gv", "--maxgvs", default=10, help="Max # of GVs")
#parser.add_argument("-nb", "--numbkg", default=-1, help="Max # of bkg tracks")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=4000, help="Evt # to end with")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

datafile  = "files/"+args.data
labelfile = "files/"+args.label

datatree = None
labeltree = None

evt_graphs = None

print("Loading files...")
with uproot.open(datafile) as f:
    demo = f['demo']
    datatree = demo['tree']

with uproot.open(labelfile) as f:
    labeltree = f['tree']

if(int(args.end) > len(datatree['trk_p'].array())):
    print("Not enough events, check data file")

trk_pt_array = datatree['trk_pt'].array()
trk_ip3d_array = datatree['trk_ip3d'].array()
trk_ip2dsig_array = datatree['trk_ip2dsig'].array()
ind_array = labeltree['ind'].array()
flag_array = labeltree['flag'].array()
seed_array = labeltree['seed_ind'].array()

def create_dataobj(datatree, labeltree, trk_features, nevts=3):

    evt_objects = []
    had_objects = []

    for evt in range(int(args.start), int(args.end)):
        #if(evt>nevts): break
        print("Processing event", evt)
        evt_features = {}
        seeds = seed_array[evt]

        for feature in trk_features:
            evt_features[feature] = (datatree[feature].array())[evt]

        #print("Making event level object....")

        fullfeatmat = np.stack([evt_features[f] for f in trk_features], axis=1)
        fullfeatmat = np.asarray(fullfeatmat)
        evtsiginds = ind_array[evt]
        nan_mask = ~np.isnan(fullfeatmat).any(axis=1)
        fullfeatmat = fullfeatmat[nan_mask]
        evtsiginds = [ind for ind in evtsiginds if nan_mask[ind]]


        evt_data = Data(
                evt=evt,
                seeds=torch.tensor(seeds, dtype=torch.int16),
                x=torch.tensor(fullfeatmat, dtype=torch.float),
                siginds=torch.tensor(evtsiginds, dtype=torch.int16)
                )

        evt_objects.append(evt_data)

        #print("Making hadron level objects....")
        
        for had in np.unique(flag_array[evt]):
            numtrks = len(trk_pt_array[evt])
            allinds = set(range(numtrks))
            inds = np.where(flag_array[evt] == had)[0]
            sig_inds = ind_array[evt][inds]
            sig_inds = [ind for ind in sig_inds if ind in seed_array[evt]] #Only keeping sigs that are also seeds
            reminds = list(allinds - set(sig_inds))
            if(len(sig_inds) < 2) : continue #Atleast two tracks associated with the hadron
            
            val_reminds = []
            for ind in reminds:
                is_valid = all(not np.isnan(evt_features[feature][ind]) for feature in trk_features)
                if is_valid:
                    val_reminds.append(ind) #Making sure there are no NaNs for bkg sampling

            bkg_inds = np.random.choice(val_reminds, size=len(sig_inds), replace=False)

            labels = np.zeros(len(sig_inds)+len(bkg_inds))

            labels[:len(sig_inds)] = 1  # First part of the array is for signal tracks

            combined_inds = np.hstack([sig_inds, bkg_inds])  # Concatenate signal and background indices
            feature_matrix = np.vstack([np.array([evt_features[feature][ind] for ind in combined_inds])
                                for feature in trk_features]).T  # Stack and transpose to get the right shape

            had_data = Data(
                    evt=evt,
                    had=had,
                    seeds = torch.tensor(sig_inds, dtype=torch.int16), #For training, seeds are sigs
                    x=torch.tensor(feature_matrix, dtype=torch.float),
                    y=torch.tensor(labels, dtype=torch.float)
                    )

            had_objects.append(had_data)

    return evt_objects, had_objects

        
print("Creating data objects...")

evt_data, had_data = create_dataobj(datatree, labeltree, trk_features)

print(f"Saving evt_data to evtdata_{args.save_tag}.pkl...")
with open("evtdata_"+args.save_tag+".pkl", 'wb') as f:
    pickle.dump(evt_data, f)

print(f"Saving had_data to haddata_{args.save_tag}.pkl...")
with open("haddata_"+args.save_tag+".pkl", 'wb') as f:
    pickle.dump(had_data, f)
