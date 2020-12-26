# Infant-movements-analysis
I implemented feature extraction defined by Tsuji et al. [1].

I used paper [1] and C++ program created by authors as a reference.

# Environments of my implementation
- OS: Windows 10
- Python: Anaconda 3 (python version: 3.8.3)
- OpenCV: 4.0.1
  - if your envirment doesn't have OpenCV, please install it by command "conda install opencv".

# How to use
1. 計測動画と背景画像をdataフォルダに入れる
2. main.pyを開いて，計測動画と背景画像のパスを関数feature_extractionのfilenameMoveiとfilenameBackImageに指定する
3. main.pyの実行
4. 結果はresultsフォルダに実行時刻が名前のフォルダに保存される

# Outputs
- body_change.csv: the changes in body posture at each time
  - A1-A9 is analysis area defined as [1]
- COG_features.csv: COG features at each time
  - G_x and G_y are coordinates of COG.
  - G^v_x and G^v_y are the velocity of COG.
  - G^d_x and G^d_y are the fluctuation of COG.
- movement_change.csv: the changes in body movement at each time
- setting.csv: settring parameters of analysis

# Etc
- The processing of the paper [1] and the program do not exactly match because the processing of the paper [1] and the C++ program do not exactly match.

# Reference
- [1] [Markerless Measurement and evaluation of General Movements in infants](https://doi.org/10.1038/s41598-020-57580-z)
