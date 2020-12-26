import cv2
import numpy as np
import pandas as pd
import math


def save_params(l_params, l_cols, savepath):
    params = pd.DataFrame([l_params])
    params.columns = l_cols
    print(params)
    params.T.to_csv(savepath + "/setting.csv", header=None)


def rotate_img_align_head_left(img, head_orient):
    """顔が画像左向きになるように画像を回転させる
    """
    if head_orient == "right":
        # 180度回転
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    elif head_orient == "above":
        # 反時計回りに90度回転
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif head_orient == "below":
        # 時計回りに90度回転
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated_img = img.copy()
    return rotated_img


# def nothing(s):
#     """空の関数

#     Arguments:
#         s {int} -- 差分画素の閾値．0〜255の値の範囲を持つ
#     """
#     pass


# def decide_threshold(img, back_img, thresh, head_direction):
#     """対話的に閾値を決める関数

#     Arguments:
#         img {numpy.float32} -- 入力画像
#         back_img {numpy.float32} -- 背景画像
#         thresh {int} -- 仮の閾値
#         head_direction {int} -- 新生児の頭の向き
#     """
#     cv2.namedWindow("image")
#     cv2.createTrackbar("threshold", "image", thresh, 255, nothing)
#     switch = "L,R,A,B"
#     cv2.createTrackbar(switch, "image", 0, 3, nothing)
#     flag = True
#     while flag:
#         diff_img = diff_threshold_process(img, back_img, thresh)
#         diff_img = denoising(diff_img, 2)
#         cv2.imshow("image", diff_img)
#         thresh = cv2.getTrackbarPos("threshold", "image")
#         head_direction = cv2.getTrackbarPos(switch, "image")
#         key = cv2.waitKey(1)
#         if key == ord("q"):
#             print(thresh)
#             print(head_direction)
#             flag = False
#             cv2.destroyAllWindows()
#             break

#     return thresh, head_direction


def diff_threshold_process(img1, img2, thresh):
    """img1とimg2の差分画像から閾値を用いて2値化画像を作成する関数
    C++版の領域依存システムと同じ処理内容です

    Arguments:
        img1 {numpy.float32} -- 入力画像1
        img2 {numpy.float32} -- 入力画像2
        thresh {int} -- 2値化の閾値

    Returns:
        numpy.float32 -- 2値画像
    """
    img = cv2.absdiff(img1, img2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img[:, :, 2]
    _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    return img


# def diff_threshold_process_(img1, img2, thresh):
#     """img1とimg2の差分画像から閾値を用いて2値化画像を作成する関数
#     グレースケールに変換してから差分画像を計算してます

#     Arguments:
#         img1 {numpy.float32} -- 入力画像1
#         img2 {numpy.float32} -- 入力画像2
#         thresh {int} -- 2値化の閾値

#     Returns:
#         numpy.float32 -- 2値画像
#     """
#     img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     img = cv2.absdiff(img1_gray, img2_gray)
#     _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

#     return img


# # NOTE: 不要
# def swap(data, head_direction, deg):
#     """頭の向きに応じて特徴量を並び替える関数

#     Arguments:
#         data {list} -- 各身体部位の特徴量
#         head_direction {str} -- 頭の向き
#         deg {float} -- 近似した楕円の傾き

#     Returns:
#         list -- 並び替えられた特徴量
#     """
#     if head_direction == "right":
#         data[0], data[3] = data[3], data[0]
#         data[1], data[2] = data[2], data[1]
#     elif head_direction == "above" and deg < 0:
#         data[0], data[3] = data[3], data[0]
#         data[1], data[2] = data[2], data[1]
#     elif head_direction == "below" and deg > 0:
#         data[0], data[3] = data[3], data[0]
#         data[1], data[2] = data[2], data[1]

#     return data


def denoising(img, N):
    """膨張・縮小によるノイズ処理関数

    Arguments:
        img {numpy.float32} -- 2値画像
        N {int} -- 膨張・縮小の回数

    Returns:
        numpy.float32 -- ノイズ処理後の画像
    """
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(N):
        img = cv2.erode(img, kernel, 1)
    for _ in range(N):
        img = cv2.dilate(img, kernel, 1)

    return img


def rotate_pts(pts, deg):
    """点のアフィン変換
    画像左上を原点として時計回りにdegだけ回転させる
    deg<0のときは反時計回りに回転する

    Arguments:
        pts {list} -- 回転させたい点群
        deg {int} -- 回転させる角度

    Returns:
        list -- アフィン変換で回転後の点群
    """
    points = np.array(
        [
            [pts[0][0], pts[0][1], 1],
            [pts[1][0], pts[1][1], 1],
            [pts[2][0], pts[2][1], 1],
            [pts[3][0], pts[3][1], 1],
        ],
        np.float64,
    )

    affine_mat = np.zeros((3, 2))
    affine_mat[0, 0] = math.cos(math.radians(deg))
    affine_mat[1, 0] = -math.sin(math.radians(deg))
    affine_mat[2, 0] = 0
    affine_mat[0, 1] = math.sin(math.radians(deg))
    affine_mat[1, 1] = math.cos(math.radians(deg))
    affine_mat[2, 1] = 0

    rotated_pts = np.dot(points, affine_mat)

    return rotated_pts


# # NOTE: 不要
# def count_pixels(img, Coe):
#     """右上半身，左上半身，右下半身，左下半身の画素値の総和を求める関数

#     Arguments:
#         img {numpy.float32} -- 2値画像
#         Coe {list} -- 身体を囲む直線の係数と身体の上下左右を分離する直線の係数

#     Returns:
#         list, list -- 各身体部位の画素値総和，xy方向の重心
#     """
#     white_pixels_each_area = [0 for _ in range(9)]
#     moment = [0, 0]

#     _diff = np.where(img == 255)
#     _x = _diff[1]
#     _y = _diff[0]
#     whether_roi1 = _diff[1] > (Coe[0][0] * _y + Coe[0][1])
#     whether_roi2 = _diff[1] < (Coe[1][0] * _y + Coe[1][1])
#     whether_roi3 = _diff[0] < (Coe[2][0] * _x + Coe[2][1])
#     whether_roi4 = _diff[0] > (Coe[3][0] * _x + Coe[3][1])
#     whether_roi1 = np.bitwise_and(whether_roi1, whether_roi2)
#     whether_roi1 = np.bitwise_and(whether_roi1, whether_roi3)
#     whether_roi1 = np.bitwise_and(whether_roi1, whether_roi4)
#     _x = _x[whether_roi1]
#     _y = _y[whether_roi1]

#     normalized_line_x = Coe[4][0] * _y + Coe[4][1]
#     normalized_line_y = Coe[5][0] * _x + Coe[5][1]
#     # print(normalized_line_x)
#     # print(normalized_line_y)

#     white_pixels_each_area[0] = np.sum(
#         np.bitwise_and(_x <= normalized_line_x, _y >= normalized_line_y)
#     )
#     white_pixels_each_area[1] = np.sum(
#         np.bitwise_and(_x <= normalized_line_x, _y < normalized_line_y)
#     )
#     white_pixels_each_area[2] = np.sum(
#         np.bitwise_and(_x > normalized_line_x, _y >= normalized_line_y)
#     )
#     white_pixels_each_area[3] = np.sum(
#         np.bitwise_and(_x > normalized_line_x, _y < normalized_line_y)
#     )
#     moment[0] = np.sum(_x)
#     moment[1] = np.sum(_y)
#     # print(white_pixels_each_area)
#     # print(moment)
#     # print()

#     # 画素値の総和を計算（gravityの計算で必要となるため）
#     # でも，総和だけ計算するのは不自然だな
#     white_pixels_each_area[8] = sum(white_pixels_each_area[0:4])

#     return white_pixels_each_area, moment


# def lpf(CutFreq, LPFSampleFreq, data):
#     """2次のバターワースローパスフィルタ

#     Arguments:
#         CutFreq {int} -- カットオフ周波数
#         LPFSampleFreq {int} -- サンプリング周波数
#         data {pandas.DataFrame} -- 入力データ

#     Returns:
#         numpy.float32 -- フィルター後のデータ
#     """
#     fs = LPFSampleFreq
#     fc = CutFreq
#     order = 2  # order of butterworth filter
#     w = fc / (fs / 2)  # Normalize the frequency
#     b, a = signal.butter(order, w, "low")
#     output = signal.lfilter(b, a, np.array(data), axis=0)
#     return output


# TODO: 関数名を再検討
def get_circumscribed_rect_vertices_of_ellipse(center, size, theta):
    """
    vertex_coors[*][0]: x座標
    vertex_coors[*][1]: y座標
    vertex_coors[0][*]: theta=0のときに左下にくる頂点座標
    vertex_coors[1][*]: theta=0のときに左上にくる頂点座標
    vertex_coors[2][*]: theta=0のときに右上にくる頂点座標
    vertex_coors[3][*]: theta=0のときに右下にくる頂点座標
    """
    c_x, c_y = center[0], center[1]
    v_sin = math.sin(math.radians(theta))
    v_cos = math.cos(math.radians(theta))
    A_x = -0.5 * size[0] * v_sin
    A_y = -0.5 * size[0] * v_cos
    B_x = 0.5 * size[1] * v_cos
    B_y = -0.5 * size[1] * v_sin
    C_x = 0.5 * size[0] * v_sin
    C_y = 0.5 * size[0] * v_cos
    D_x = -0.5 * size[1] * v_cos
    D_y = 0.5 * size[1] * v_sin
    vertex_coors = []
    vertex_coors.append([c_x + A_x + D_x, c_y + A_y + D_y])
    vertex_coors.append([c_x + A_x + B_x, c_y + A_y + B_y])
    vertex_coors.append([c_x + B_x + C_x, c_y + B_y + C_y])
    vertex_coors.append([c_x + C_x + D_x, c_y + C_y + D_y])
    return vertex_coors


def get_vertices_expanded_rect(rotated_vertex_coors, t_a1, t_a2, t_a3, W, H):
    """
    vertex_coors[*][0]: x座標
    vertex_coors[*][1]: y座標
    """
    a1 = t_a1 * W
    a2 = t_a2 * W
    a3 = t_a3 * H
    vertex_coors = [[0, 0] for tmp in range(4)]
    vertex_coors[0][0] = rotated_vertex_coors[0][0] - a1
    vertex_coors[0][1] = rotated_vertex_coors[0][1] - a3
    vertex_coors[1][0] = rotated_vertex_coors[1][0] + a2
    vertex_coors[1][1] = rotated_vertex_coors[1][1] - a3
    vertex_coors[2][0] = rotated_vertex_coors[2][0] + a2
    vertex_coors[2][1] = rotated_vertex_coors[2][1] + a3
    vertex_coors[3][0] = rotated_vertex_coors[3][0] - a1
    vertex_coors[3][1] = rotated_vertex_coors[3][1] + a3
    return vertex_coors


def divide_area_into_four(vertex_coors, gamma, delta):
    # 全身の領域を４分割する線の交点座標
    norm_x = vertex_coors[0][0] + gamma * \
        abs(vertex_coors[1][0] - vertex_coors[0][0])
    norm_y = vertex_coors[0][1] + delta * \
        abs(vertex_coors[3][1] - vertex_coors[0][1])
    # 論文で言うところのU-U'とV-V'を求めている
    # baseline[0][*]: 全身を上・下半身に分割する線とA9領域矩形の上側との交点
    # baseline[1][*]: 全身を左・右半身に分割する線とA9領域矩形の右側との交点
    # baseline[2][*]: 全身を上・下半身に分割する線とA9領域矩形の下側との交点
    # baseline[3][*]: 全身を左・右半身に分割する線とA9領域矩形の左側との交点
    baseline = [[0, 0] for tmp in range(4)]
    baseline[0][0] = norm_x
    baseline[0][1] = vertex_coors[0][1]
    baseline[1][0] = vertex_coors[1][0]
    baseline[1][1] = norm_y
    baseline[2][0] = norm_x
    baseline[2][1] = vertex_coors[2][1]
    baseline[3][0] = vertex_coors[3][0]
    baseline[3][1] = norm_y
    return baseline


def get_contour_maximize_rect_area(contours):
    if len(contours) > 1:
        area_max = 0
        i_max = 0
        for i in range(len(contours)):
            x_, y_, w_, h_ = cv2.boundingRect(contours[i])
            area = w_ * h_
            if area_max < area:
                area_max = area
                i_max = i
        cnt = contours[i_max]
    else:
        cnt = contours[0]
    return cnt


def display_images(dispImg, diff_img, interframe_diff, pts,
                   affine_pts, affine_pts_baseline,
                   affine_pts_margins, gravity):
    _ = np.array(
        (
            (int(affine_pts_margins[0][0]),
             int(affine_pts_margins[0][1])),
            (int(affine_pts_margins[1][0]),
             int(affine_pts_margins[1][1])),
            (int(affine_pts_margins[2][0]),
             int(affine_pts_margins[2][1])),
            (int(affine_pts_margins[3][0]),
             int(affine_pts_margins[3][1])),
        )
    )
    cv2.polylines(dispImg, [_], True, (255, 255, 255), thickness=2)
    cv2.polylines(diff_img, [_], True, (255, 255, 255), thickness=2)
    cv2.polylines(interframe_diff, [_], True, (255, 255, 255), thickness=2)

    p1 = (int(affine_pts_baseline[0][0]), int(affine_pts_baseline[0][1]))
    p2 = (int(affine_pts_baseline[2][0]), int(affine_pts_baseline[2][1]))
    cv2.line(dispImg, p1, p2, (255, 255, 255), thickness=1, lineType=cv2.LINE_4)
    cv2.line(diff_img, p1, p2, (255, 255, 255), thickness=1, lineType=cv2.LINE_4)
    cv2.line(interframe_diff, p1, p2, (255, 255, 255), thickness=1, lineType=cv2.LINE_4)

    p1 = (int(affine_pts_baseline[1][0]), int(affine_pts_baseline[1][1]))
    p2 = (int(affine_pts_baseline[3][0]), int(affine_pts_baseline[3][1]))
    cv2.line(dispImg, p1, p2, (255, 255, 255), thickness=1, lineType=cv2.LINE_4)
    cv2.line(diff_img, p1, p2, (255, 255, 255), thickness=1, lineType=cv2.LINE_4)
    cv2.line(interframe_diff, p1, p2, (255, 255, 255), thickness=1, lineType=cv2.LINE_4)

    # Drawing coordinates of COG
    cv2.circle(dispImg, (int(
        gravity[-1][0]), int(gravity[-1][1])), 10, (0, 0, 125), thickness=-1)

    cv2.imshow("Measured video", dispImg)
    cv2.imshow("diff img", diff_img)
    cv2.imshow("frame diff img", interframe_diff)


def calc_slopes_intercepts_lines_around_rect_A9(cortex_coors, preframe_vals):
    slopes_interpects = []

    # 拡大した外接矩形の頭側の辺の傾きと切片を求める
    _ = calc_slope_intercept_x_my_n(
        cortex_coors[0], cortex_coors[3], preframe_vals[0])
    slopes_interpects.append(_)

    # 脚側
    _ = calc_slope_intercept_x_my_n(
        cortex_coors[1], cortex_coors[2], preframe_vals[1])
    slopes_interpects.append(_)

    # 身体右側
    _ = calc_slope_intercept_y_px_q(
        cortex_coors[2], cortex_coors[3], preframe_vals[2])
    slopes_interpects.append(_)

    # 身体左側
    _ = calc_slope_intercept_y_px_q(
        cortex_coors[0], cortex_coors[1], preframe_vals[3])
    slopes_interpects.append(_)

    return slopes_interpects


def calc_slopes_intercepts_lines_UU_VV(pts_coors, preframe_vals):
    slopes_intercepts = []

    # 身体を上半身・下半身に分割する線U-U': x=my+nの傾きmと切片nを計算
    _ = calc_slope_intercept_x_my_n(
        pts_coors[0], pts_coors[2], preframe_vals[0])
    slopes_intercepts.append(_)

    # 身体を左半身・右半身に分割する線V-V': y=px+qの傾きpと切片qを計算
    _ = calc_slope_intercept_y_px_q(
        pts_coors[1], pts_coors[3], preframe_vals[1])
    slopes_intercepts.append(_)

    return slopes_intercepts


def calc_slope_intercept_y_px_q(pts1, pts2, preframe_val):
    """点1 (pts1)と点2 (pts2)を結ぶ直線y=px+qの
    傾きpと切片qを計算する
    """
    p1_x, p1_y = pts1[0], pts1[1]
    p2_x, p2_y = pts2[0], pts2[1]
    EPSIRON = 10 ** (-6)
    if abs(p1_x - p2_x) > EPSIRON:
        slope = (p1_y - p2_y) / (p1_x - p2_x)
        intercept = (p1_x * p2_y - p1_y * p2_x) / (p1_x - p2_x)
    else:
        slope = preframe_val[0]
        intercept = preframe_val[1]
    return [slope, intercept]


def calc_slope_intercept_x_my_n(pts1, pts2, preframe_val):
    """点1 (pts1)と点2 (pts2)を結ぶ直線x=my+nの
    傾きmと切片nを計算する
    """
    p1_x, p1_y = pts1[0], pts1[1]
    p2_x, p2_y = pts2[0], pts2[1]
    EPSIRON = 10 ** (-6)
    if abs(p1_y - p2_y) > EPSIRON:
        slope = (p1_x - p2_x) / (p1_y - p2_y)
        intercept = (p1_y * p2_x - p1_x * p2_y) / (p1_y - p2_y)
    else:
        slope = preframe_val[0]
        intercept = preframe_val[1]
    return [slope, intercept]


def sum_white_pixels_calc_cog(diff_img, l_slp_intp_A9, l_slp_intp_UU_VV):
    coors_white_pixel = np.where(diff_img == 255)
    coors_x = coors_white_pixel[1]
    coors_y = coors_white_pixel[0]

    # TODO: 関数化して領域内のcoors_xとcoors_yを受け取る
    cond1 = coors_x > (l_slp_intp_A9[0][0] * coors_y + l_slp_intp_A9[0][1])
    cond2 = coors_x < (l_slp_intp_A9[1][0] * coors_y + l_slp_intp_A9[1][1])
    cond3 = coors_y < (l_slp_intp_A9[2][0] * coors_x + l_slp_intp_A9[2][1])
    cond4 = coors_y > (l_slp_intp_A9[3][0] * coors_x + l_slp_intp_A9[3][1])

    is_inside_A9 = np.bitwise_and(cond1, cond2)
    is_inside_A9 = np.bitwise_and(is_inside_A9, cond3)
    is_inside_A9 = np.bitwise_and(is_inside_A9, cond4)

    coors_x = coors_x[is_inside_A9]
    coors_y = coors_y[is_inside_A9]

    # 領域内の白画素の座標値を直線U-U'，V-V'上に射影
    coors_x_line_UU = l_slp_intp_UU_VV[0][0] * coors_y + l_slp_intp_UU_VV[0][1]
    coors_y_line_VV = l_slp_intp_UU_VV[1][0] * coors_x + l_slp_intp_UU_VV[1][1]

    l_total_white_pixels = [0 for i in range(9)]
    # 領域A1
    l_total_white_pixels[0] = np.sum(np.bitwise_and(
        coors_x <= coors_x_line_UU, coors_y < coors_y_line_VV))
    # A2
    l_total_white_pixels[1] = np.sum(np.bitwise_and(
        coors_x <= coors_x_line_UU, coors_y >= coors_y_line_VV))
    # A3
    l_total_white_pixels[2] = np.sum(np.bitwise_and(
        coors_x > coors_x_line_UU, coors_y < coors_y_line_VV))
    # A4
    l_total_white_pixels[3] = np.sum(np.bitwise_and(
        coors_x > coors_x_line_UU, coors_y >= coors_y_line_VV))
    # A9
    l_total_white_pixels[8] = sum(l_total_white_pixels[0:4])

    # Center of Gravity (COG)の計算
    if l_total_white_pixels[8] == 0:
        center_of_gravity = [0, 0]
    else:
        cog_x = np.sum(coors_x) / l_total_white_pixels[8]
        cog_y = np.sum(coors_y) / l_total_white_pixels[8]
        center_of_gravity = [cog_x, cog_y]
    return l_total_white_pixels, center_of_gravity


def calc_from_A5_to_A8(body_chg_, movt_chg_):
    body_chg = body_chg_.copy()
    movt_chg = movt_chg_.copy()
    # 領域A5(=A1+A2)
    body_chg.iloc[:, 4] = body_chg.iloc[:, 0] + body_chg.iloc[:, 1]
    movt_chg.iloc[:, 4] = movt_chg.iloc[:, 0] + movt_chg.iloc[:, 1]
    # 領域A6(=A3+A4)
    body_chg.iloc[:, 5] = body_chg.iloc[:, 2] + body_chg.iloc[:, 3]
    movt_chg.iloc[:, 5] = movt_chg.iloc[:, 2] + movt_chg.iloc[:, 3]
    # 領域A7(=A1+A3)
    body_chg.iloc[:, 6] = body_chg.iloc[:, 0] + body_chg.iloc[:, 2]
    movt_chg.iloc[:, 6] = movt_chg.iloc[:, 0] + movt_chg.iloc[:, 2]
    # 領域A8(=A2+A4)
    body_chg.iloc[:, 7] = body_chg.iloc[:, 1] + body_chg.iloc[:, 3]
    movt_chg.iloc[:, 7] = movt_chg.iloc[:, 1] + movt_chg.iloc[:, 3]
    return body_chg, movt_chg


def calc_COG_features(COG, W_avg, body_area, Fs):
    # COG velocity
    COG_vel = [[0, 0]]
    for i in range(1, COG.shape[0]):
        COG_v_x = COG.iloc[i, 0] - COG.iloc[i - 1, 0]
        COG_v_y = COG.iloc[i, 1] - COG.iloc[i - 1, 1]
        COG_vel.append([COG_v_x, COG_v_y])
    COG_vel = Fs * pd.DataFrame(COG_vel) / np.sqrt(body_area)

    # COG fluctuation
    COG_avg = COG[0:W_avg].mean(axis=0).to_list()
    COG_flctn_x = (COG.iloc[:, 0] - COG_avg[0]) / np.sqrt(body_area)
    COG_flctn_y = (COG.iloc[:, 1] - COG_avg[1]) / np.sqrt(body_area)
    COG_flctn = pd.concat([COG_flctn_x, COG_flctn_y], axis=1)

    COG_features = pd.concat([COG, COG_vel], axis=1)
    COG_features = pd.concat([COG_features, COG_flctn], axis=1)
    return COG_features


def save_features(body_chg, movt_chg, COG, savepath):
    # add column name
    body_chg.columns = ["A" + str(i) for i in range(1, 10)]
    movt_chg.columns = ["A" + str(i) for i in range(1, 10)]
    COG.columns = ["G_x", "G_y", "G^v_x", "G^v_y", "G^d_x", "G^d_y"]

    # save
    path_body_chg = savepath + "/body_change.csv"
    path_movt_chg = savepath + "/movement_change.csv"
    path_cog = savepath + "/COG_features.csv"
    body_chg.to_csv(path_body_chg, header=True, index=None)
    movt_chg.to_csv(path_movt_chg, header=True, index=None)
    COG.to_csv(path_cog, header=True, index=None)
