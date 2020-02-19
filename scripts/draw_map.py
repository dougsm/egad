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

# Global diversity
# special = {'(0, 0)': ['0124_12431'], '(0, 3)': ['0596_7769'], '(1, 3)': ['0767_11744'], '(0, 4)': ['0752_10228'], '(0, 2)': ['0201_20139'], '(0, 5)': ['0626_10791'], '(0, 1)': ['0654_0407'], '(1, 0)': ['0423_13766'], '(1, 1)': ['0548_2990'], '(2, 0)': ['0636_11793'], '(1, 2)': ['0455_16969'], '(1, 6)': ['0259_25948'], '(0, 6)': ['0477_19143'], '(4, 4)': ['0869_1966'], '(6, 0)': ['0150_15090'], '(2, 1)': ['0717_6732'], '(3, 1)': ['0617_9851'], '(1, 4)': ['0771_12189'], '(2, 3)': ['0239_23976'], '(3, 5)': ['0203_20367'], '(2, 2)': ['0896_4666'], '(3, 2)': ['0566_4788'], '(3, 3)': ['0414_12899'], '(1, 5)': ['0652_0274'], '(4, 5)': ['0993_7376'], '(3, 6)': ['0271_27169'], '(5, 2)': ['0587_6880'], '(2, 5)': ['0309_2325'], '(2, 6)': ['0894_4492'], '(6, 4)': ['0468_18280'], '(4, 3)': ['0613_9488'], '(5, 3)': ['0645_12678'], '(3, 0)': ['0259_25985'], '(5, 4)': ['0971_5192'], '(6, 5)': ['0276_27628'], '(5, 5)': ['0383_9785'], '(2, 4)': ['0850_0053'], '(4, 6)': ['0754_10450'], '(5, 1)': ['0401_11593'], '(4, 2)': ['0562_4326'], '(5, 6)': ['0929_0949'], '(6, 1)': ['0152_15219'], '(6, 2)': ['0871_2102'], '(3, 4)': ['0630_11106'], '(4, 1)': ['0103_10359'], '(4, 0)': ['0531_1217'], '(6, 3)': ['0348_6272'], '(5, 0)': ['0805_15591'], '(6, 6)': ['0451_16569']}

# Local diversity
special ={
    "(0, 0)": [
        "0102_10261"
    ],
    "(0, 1)": [
        "1419_0067"
    ],
    "(0, 2)": [
        "0412_12669"
    ],
    "(0, 3)": [
        "1393_18321"
    ],
    "(0, 4)": [
        "0300_1415"
    ],
    "(0, 5)": [
        "0626_10791"
    ],
    "(0, 6)": [
        "0960_4012"
    ],
    "(1, 0)": [
        "1389_17989"
    ],
    "(1, 1)": [
        "0530_1191"
    ],
    "(1, 2)": [
        "1872_1412"
    ],
    "(1, 3)": [
        "0935_1514"
    ],
    "(1, 4)": [
        "1379_16945"
    ],
    "(1, 5)": [
        "1315_10582"
    ],
    "(1, 6)": [
        "0259_25948"
    ],
    "(2, 0)": [
        "1130_13051"
    ],
    "(2, 1)": [
        "1857_21947"
    ],
    "(2, 2)": [
        "1030_3100"
    ],
    "(2, 3)": [
        "1962_10423"
    ],
    "(2, 4)": [
        "1345_13544"
    ],
    "(2, 5)": [
        "0977_5732"
    ],
    "(2, 6)": [
        "0430_14497"
    ],
    "(3, 0)": [
        "0636_11793"
    ],
    "(3, 1)": [
        "0817_16750"
    ],
    "(3, 2)": [
        "1421_0295"
    ],
    "(3, 3)": [
        "0486_20066"
    ],
    "(3, 4)": [
        "1861_0392"
    ],
    "(3, 5)": [
        "1085_8531"
    ],
    "(3, 6)": [
        "0513_22712"
    ],
    "(4, 0)": [
        "0116_11659"
    ],
    "(4, 1)": [
        "1885_2788"
    ],
    "(4, 2)": [
        "0293_0755"
    ],
    "(4, 3)": [
        "0613_9488"
    ],
    "(4, 4)": [
        "0257_25775"
    ],
    "(4, 5)": [
        "1868_1059"
    ],
    "(4, 6)": [
        "1296_8637"
    ],
    "(5, 0)": [
        "0152_15240"
    ],
    "(5, 1)": [
        "1096_9699"
    ],
    "(5, 2)": [
        "1880_2219"
    ],
    "(5, 3)": [
        "1627_20815"
    ],
    "(5, 4)": [
        "0993_7376"
    ],
    "(5, 5)": [
        "0754_10450"
    ],
    "(5, 6)": [
        "0430_14482"
    ],
    "(6, 0)": [
        "0064_6425"
    ],
    "(6, 1)": [
        "0490_20422"
    ],
    "(6, 2)": [
        "0055_5573"
    ],
    "(6, 3)": [
        "0031_3133"
    ],
    "(6, 4)": [
        "1819_18011"
    ],
    "(6, 5)": [
        "1991_13378"
    ],
    "(6, 6)": [
        "1600_18195"
    ]
}

special = [vv[0] for vv in special.values()]


p = Path(sys.argv[1]) / 'pool'
po = p.parent
print(po)

gen_output = sorted(po.glob('gen_[0-9]*.png'))

DIM1 = 25
DIM2 = 25

new_map_size = (7, 7)
k1_mm = (2, 23)  # Complexity
k2_mm = (1, 22)  # Difficult (0 == hard)

cov_in = 0
cov_out = 0

while True:
    gens = sorted(po.glob('gen_[0-9]*.json'))

    lg = len(gens)
    lgo = len(gen_output)

    if lg > lgo:
        time.sleep(0.5)

        d = lgo - lg
        with open(gens[d]) as f:
            gen_mshs = json.load(f)

        n = gens[d].name
        n = n.replace('gen_', 'gen_div_')
        div_p = gens[d].parent / n
        if div_p.exists():
            with open(div_p) as f:
                divs = json.load(f)
        else:
            divs = {}

        drwg = []
        for gd in range(DIM2):
            for inds in [(0, 1), (2, 3)]:
                nhr = []
                for nh in range(DIM1):
                    k = str((nh, gd))
                    for i in inds:
                        try:
                            img = imageio.imread(p / '{}.png'.format(gen_mshs[k][i]))
                            # if divs:
                            #     bc = (np.array(scalar.to_rgba(divs.get(gen_mshs[k][i], 0.0)))*255.0).astype(np.uint8)
                            #     img[img[:, :, :3].min(axis=2) == 255] = bc

                                # nk1 = int(((nh + (0.5 if i in (1,3) else 0.0) - k1_mm[0]) / (k1_mm[1] - k1_mm[0]) * new_map_size[0] - 1e-6))
                                # nk2 = int(((gd + (0.5 if i in (2,3) else 0.0) - k2_mm[0]) / (k2_mm[1] - k2_mm[0]) * new_map_size[1] - 1e-6))
                                #
                                # nk1 = max(min(nk1, new_map_size[0]-1), 0)
                                # #nk2 = min(nk2, new_map_size-1)
                                #
                                # if gen_mshs[k][i] in special or nk1 in range(new_map_size[0]) and nk2 in range(new_map_size[1]):
                                #     if gen_mshs[k][i] in special:
                                #         bc = [255, 0, 0, 255]
                                #     else:
                                #         bc = (np.array(scalar.to_rgba(nk1+nk2))*255.0).astype(np.uint8)
                                #     img[img[:,:,:3].min(axis=2)==255] = bc

                            nhr.append(img)
                            cov_in += 1
                        except (KeyError, IndexError):
                            cov_out += 1
                            nhr.append(np.ones((64, 64, 4), dtype=np.uint8)*255)
                drwg.append(nhr)

        img = np.vstack([np.hstack(dd) for dd in drwg])

        o = gens[d].with_suffix('.png')
        imageio.imsave(o, img[:, :, :3])
        imageio.imsave(po/'curr.png', img[:, :, :3])

        gen_output.append(o)
        print(cov_in, cov_out, cov_in/(2500))
        print(str(o))

        # exit()

    else:
        time.sleep(10)
