import os
import datetime
import feature_extraction as ext


# create folder to save output
dt_now = datetime.datetime.now()
savepath = "./results/" + str(dt_now)[:-7]
savepath = savepath.replace(" ", "_")
savepath = savepath.replace(":", "-")
os.mkdir(savepath)

# 0: left, 1: right, 2: above, 3: below
head_orientation = 0

# threshold for binalization
thresh = 20

ext.feature_extraction(
    filenameMovie="./data/test.mp4",
    filenameBackImage="./data/back_test.bmp",
    head_orientation=head_orientation,
    threshold=thresh,
    isDispMovie=True,
    savepath=savepath,
)
