import torch
import argparse
import pickle

parser = argparse.ArgumentParser("Testing seed data")

parser.add_argument("-l", "--load", default="", help="Path of datafile to load")

args = parser.parse_args()

graph_data = []

if args.load != "":
    print(f"Loading seed data from {args.load}...")
    with open(args.load, 'rb') as f:
        evt_data = pickle.load(f)

for i, data in enumerate(evt_data):
    if(i==10):break
    print("EVENT", i)

    #print("LABELS", data.y)
    
    print("DATA", data.x)
    #print("SEEDS", data.seeds)
    #print("SIGINDS", data.siginds)
    #print("SIGFLAGS", data.sigflags)
    print("LABELS", data.y)
    print("EDGE INDEX", data.edge_index)
    print("EDGE ATTR", data.edge_attr)
    #print("SVINDS", data.svinds)
    
    #print("Total tracks: ", data.x.size(0))
    #seed_tracks = data.seeds.nonzero(as_tuple=True)[0]
    #sig_tracks = (data.y.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    #print("Seed tracks", seed_tracks.size(0))
    #print("Signal tracks", sig_tracks.size(0))
    #
    #overlap_mask = torch.isin(seed_tracks, sig_tracks)
    #overlap_tracks = seed_tracks[overlap_mask]
    #overlap_count = overlap_tracks.size(0)

    #print(f"Number of overlapping tracks between seeds and signal: {overlap_count}")
    print("NEXT")

#tottrks = 0
#seedtrks = 0
#for i, data in enumerate(evt_data):
#    tottrks += data.seeds.size(0)
#    seedtrks += torch.sum(data.seeds == 1).item()
#
#avgtot = tottrks/len(evt_data)
#avgseed = seedtrks/len(evt_data)
#print(f"Avg Number of seeds tracks over average number of tracks: {avgseed}/{avgtot}")
    
    
