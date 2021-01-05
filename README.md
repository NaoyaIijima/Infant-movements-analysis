# Infant Movements Analysis
I implemented feature extraction defined by Tsuji et al. [1].

I used paper [1] and original program (C++) created by authors as a reference.

# Environments of my implementation
- OS: Windows 10
- Python: Anaconda 3 (Python version: 3.8.3)
- OpenCV: 4.0.1
  - if your envirment doesn't have OpenCV, please install it by command "conda install opencv".

# How to use
1. Put the measured video and background image to the "data" folder.
2. Open main.py, specify path of the measured video and background image to filenameMovie and filenameBackImage that are args of function "feature_extraction".
3. Implement main.py.
4. Results are saved to the "results" folder that named start time of analysis.

# Outputs
- body_change.csv is the changes in body posture at each time.
  - A1-A9 is analysis area defined as [1].
- COG_features.csv is COG features at each time.
  - G_x and G_y are coordinates of COG.
  - G^v_x and G^v_y are the velocity of COG.
  - G^d_x and G^d_y are the fluctuation of COG.
- movement_change.csv is the changes in body movement at each time.
- setting.csv is setting parameters of analysis.

# Etc
- The processing of the paper [1] and the program do not exactly match because the processing of the paper [1] and the original program do not exactly match.

# Reference
[1] [Markerless Measurement and evaluation of General Movements in infants](https://doi.org/10.1038/s41598-020-57580-z)
