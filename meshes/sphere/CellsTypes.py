# Count the number of triangls, quads, pentagons in each case
import json
import glob
import re
import numpy as np
for file in glob.glob('./*.json'):
    meshN = re.search(r'(\d+)_circle\.json',file).group(1)
    with open(file,'r') as fh:
        data = json.load(fh)
    eC = np.zeros(10)
    bC = 0
    for cell in data['Cells'][2]:
        if (cell['Ref_elem'] != None):
            bC += 1
        else:
            eC[len(cell['Boundary'])] += 1
    print("{}_circle:".format(meshN))
    print("Boundary cells: {}".format(bC))
    print(eC)
    print("")
