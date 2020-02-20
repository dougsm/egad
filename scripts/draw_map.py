import imageio
import numpy as np
import json
from pathlib import Path
import time

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

cmap = plt.get_cmap('viridis')
norm = mpl.colors.Normalize(vmin=0.0, vmax=0.275)
scalar = cm.ScalarMappable(norm=norm, cmap=cmap)

in_path = Path(sys.argv[1])
print(in_path)
pool_path = in_path / 'pool'
output_path = in_path / 'viz'
output_path.mkdir(exist_ok=True)

existing_images = set(sorted(output_path.glob('gen_[0-9]*.png')))

DIM1 = 25
DIM2 = 25

while True:
    gens = sorted(in_path.glob('gen_[0-9]*.json'))

    for g in gens:
        if not (output_path / g.with_suffix('.png').name).exists():

            with g.open() as f:
                gen_mshs = json.load(f)

            # Load diversity values
            n = g.name
            n = n.replace('gen_', 'gen_div_')
            div_p = in_path / 'diversity' / n
            if div_p.exists():
                with open(div_p) as f:
                    divs = json.load(f)
            else:
                divs = {}

            # Create drawing from thumbnails
            drwg = []
            for gd in range(DIM2):
                for inds in [(0, 1), (2, 3)]:
                    nhr = []
                    for nh in range(DIM1):
                        k = str((nh, gd))
                        for i in inds:
                            try:
                                img = imageio.imread(pool_path / '{}.png'.format(gen_mshs[k][i]))
                                if divs:
                                    bc = (np.array(scalar.to_rgba(divs.get(gen_mshs[k][i], 0.0)))*255.0).astype(np.uint8)
                                    img[img[:, :, :3].min(axis=2) == 255] = bc

                                nhr.append(img)
                            except (KeyError, IndexError):
                                nhr.append(np.ones((64, 64, 4), dtype=np.uint8)*255)
                    drwg.append(nhr)

            img = np.vstack([np.hstack(dd) for dd in drwg])

            o = output_path / g.with_suffix('.png').name
            imageio.imsave(o, img[:, :, :3])
            imageio.imsave(output_path / 'curr.png', img[:, :, :3])

            existing_images.add(o)
            print(str(o))


    else:
        time.sleep(10)
