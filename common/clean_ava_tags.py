import os
import config
import pandas as pd

"""
This is only used to clean the AVA.txt so it doesn't contain images not seen
"""

ava_frame = pd.read_csv(config.AVA_QUALITY_LABELS_FILE, sep=" ", header=None)

f = open(config.AVA_QUALITY_LABELS_FILE)
n = open("/media/matt/New Volume/ava/cleanedlabels.txt", "w")
for idx, line in enumerate(f):
    # print(line)
    # print(idx)
    # print(ava_frame.iloc[idx, 0])
    img_name = os.path.join(config.AVA_IMAGE_PATH, str(ava_frame.iloc[idx, 0]) + '.jpg')
    if not os.path.isfile(img_name):
        continue
    n.write(line)