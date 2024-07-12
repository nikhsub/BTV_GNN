import json

trkflag_file = 'trkflags.json'

with open(trkflag_file, 'r') as f:
    data = json.load(f)

sigflags = data[0]
bkgflags = data[1]

print(sigflags['0'])
