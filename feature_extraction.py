# -*- coding: utf-8 -*-
import cv2
import sys
import pandas as pd
import utils_extract_features as util


def feature_extraction(
    filenameMovie,
    filenameBackImage,
    head_orientation=0,
    threshold=50,
    isDispMovie=False,
    savepath=None,
):
    """Function to extract infant movements features

    Arguments:
        filenameMovie {str} -- path of input video
        filenameBackMovie {str} -- path of background image

    Keyword Arguments:
        head_orientation {int} -- head orientation in image (default: {0})
        threshold {int} -- threshold for binalization (default: {50})
        isDispMovie {bool} --  Whether to display an analysis videos(default: {False})
        savepath {str} -- path to save analysis parameters (default: {None})
    """

    # dict for head orientaion
    head = {0: "left", 1: "right", 2: "above", 3: "below"}

    cap = cv2.VideoCapture(filenameMovie)
    frames = int(cap.get(7))
    resolution = str(int(cap.get(3))) + "x" + str(int(cap.get(4)))
    Fs = int(cap.get(5))  # Frame rate [fps]

    back_img = cv2.imread(filenameBackImage)
    back_img = util.rotate_img_align_head_left(back_img, head[head_orientation])

    # threshold for binalization
    thresh = threshold

    # setting coefs of margins
    Mleft = 0.3  # head (Tsuji et al., 2020 defined as t_a1)
    Mright = 0.3  # legs (Tsuji et al., 2020 defined as t_a2)
    Mtop_bottom = 0.3  # left-right (Tsuji et al., 2020 defined as t_a3)

    gamma = 0.55
    delta = 0.5

    # save parameters
    if savepath is None:
        print("[WARNING] specify savepath!!!")
        savepath = "results/temp"
    l_params = [filenameMovie, filenameBackImage, thresh,
                head_orientation, resolution, cap.get(5),
                Mleft, Mright, Mtop_bottom,
                gamma, delta]
    l_cols = ["path of input video", "path of background image", "threshold",
              "head orientation", "resolution", "frame rate",
              "coef of margin left", "coef of margin right", 
              "coef of margin top/bottom", "gamma", "delta"]
    util.save_params(l_params, l_cols, savepath)

    # list to store each feature
    body_chg = []
    movt_chg = [[0 for _ in range(9)]]
    l_COG = []

    start = 0
    end = frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frame in range(start, end):
        # Display progress
        sys.stdout.write(
            "\r" + str(round(100 * (frame - start) / (end - start), 1)) + "%")
        sys.stdout.flush()

        _, img = cap.read()
        img = util.rotate_img_align_head_left(img, head[head_orientation])

        diff_img = util.diff_threshold_process(img, back_img, thresh)
        diff_img = util.denoising(diff_img, 2)

        # NOTE: Returns of "findContours" are 2 or 3 depending version of your OpenCV.
        # # Returns are 2：
        contours, hierarchy = cv2.findContours(
            diff_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # # Returns are 3：
        # _, contours, hierarchy = cv2.findContours(
        #     diff_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )

        # Find contours to maximize area of approximate ellipse
        cnt = util.get_contour_maximize_rect_area(contours)

        ellipse = cv2.fitEllipse(cnt)

        # coordinates of center of ellipse
        # center[0]: x-axis, center[1]: y-axis
        center = ellipse[0]

        # size[0] is length of shorter axis of ellipse when deg is 0.
        # size[1] is length of longer axis of ellipse when deg is 0.
        size = ellipse[1]

        # 近似した楕円を反時計回りにellipse[2] [deg]ほど回転させると
        # 楕円の長軸がy軸（画像の縦方向）と平行になる
        # 値の範囲は0 <= deg < 180だと思われる
        deg = ellipse[2]

        # convert to -90 < deg <= 90 to simplify later calculations
        deg = 90 - deg

        # 楕円に外接する矩形の頂点座標ptsを求める
        pts = util.get_circumscribed_rect_vertices_of_ellipse(
            center, size, deg)

        # 外接矩形の短辺（楕円の短軸）がy軸と平行になるように矩形を回転
        # affine_pts[*][0]: x-axis, affine_pts[*][1]: y-axis
        # affine_pts[0][*]: 左下の頂点座標, affine_pts[1][*]: 左上の頂点座標
        # affine_pts[2][*]: 右上の頂点座標, affine_pts[3][*]: 右下の頂点座標
        affine_pts = util.rotate_pts(pts, deg)

        # 楕円の長軸の長さ（楕円の外接矩形の長辺の長さ）
        ellipseWidth = abs(affine_pts[1, 0] - affine_pts[0, 0])
        # 楕円の短軸の長さ（楕円の外接矩形の短辺の長さ）
        ellipseHeight = abs(affine_pts[3, 1] - affine_pts[0, 1])

        margins = util.get_vertices_expanded_rect(
            affine_pts, Mleft, Mright, Mtop_bottom, ellipseWidth, ellipseHeight)

        # 先ほどアフィン変換で回転した角度分を元に戻す処理
        affine_pts_margins = util.rotate_pts(margins, -deg)

        baseline = util.divide_area_into_four(margins, gamma, delta)

        # 先ほどアフィン変換で回転した角度分を元に戻す処理
        affine_pts_baseline = util.rotate_pts(baseline, -deg)

        if frame == start:
            pre_l_m = [[0, 0] for tmp_ in range(4)]
            pre_l_b = [[0, 0] for tmp_ in range(2)]
        # 全身領域A9の周囲の辺の傾きと切片を算出
        l_slope_intercept_m = util.calc_slopes_intercepts_lines_around_rect_A9(
            affine_pts_margins, pre_l_m)
        pre_l_m = l_slope_intercept_m
        # 全身領域A9を4分割する直線U-U'とV-V'の傾きと切片を算出
        l_slope_intercept_b = util.calc_slopes_intercepts_lines_UU_VV(
            affine_pts_baseline, pre_l_b)
        pre_l_b = l_slope_intercept_b

        # 領域A1-A4およびA9の内側の白画素の合計値とCOGを計算
        l_total_white_pixels, cog = util.sum_white_pixels_calc_cog(
            diff_img, l_slope_intercept_m, l_slope_intercept_b)

        body_chg.append(l_total_white_pixels)
        l_COG.append(cog)

        # We can't calculate interframe difference image in the first frame.
        if frame == start:
            preframe_img = img
            continue

        # Interframe difference
        interframe_diff = util.diff_threshold_process(
            img, preframe_img, thresh)
        interframe_diff = util.denoising(interframe_diff, 1)

        # 領域A1-A4およびA9の白画素の合計値とCOGを計算
        l_total_white_pixels, _ = util.sum_white_pixels_calc_cog(
            interframe_diff, l_slope_intercept_m, l_slope_intercept_b)

        movt_chg.append(l_total_white_pixels)
        preframe_img = img

        if isDispMovie is True:
            util.display_images(img.copy(), diff_img, interframe_diff, pts,
                                affine_pts, affine_pts_baseline,
                                affine_pts_margins, l_COG)
            key = cv2.waitKey(30)
            if key == ord("q"):
                break

    # Release
    cap.release()
    cv2.destroyAllWindows()

    body_chg = pd.DataFrame(body_chg)
    movt_chg = pd.DataFrame(movt_chg)
    body_chg, movt_chg = util.calc_from_A5_to_A8(body_chg, movt_chg)

    # 児の身体の大きさを算出
    # 初期W_avg [frame]の全身の体位変化を抽出し，昇順にして大きい方からE個の平均を算出
    E = 10
    W_avg = 30
    data_W_avg = body_chg.iloc[0:W_avg, 8]
    data_W_avg = data_W_avg.sort_values(ascending=False)
    BODY_SIZE = data_W_avg.iloc[0:E].mean(axis=0)

    # 運動変化を身体の大きさで正規化
    movt_chg = movt_chg / BODY_SIZE

    # 重心特徴量の算出
    COG = pd.DataFrame(l_COG)
    COG = util.calc_COG_features(COG, W_avg, BODY_SIZE, Fs)

    util.save_features(body_chg, movt_chg, COG, savepath)
