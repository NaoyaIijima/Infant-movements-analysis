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
head_orientation = 1

# threshold for binalization
thresh = 80

ext.feature_extraction(
    filenameMovie="./data/input.mp4",
    filenameBackImage="./data/background.bmp",
    head_orientation=head_orientation,
    threshold=thresh,
    isDispMovie=False,
    savepath=savepath,
)
