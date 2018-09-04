import cv2
import MT2Python
import copy
from scipy.optimize import leastsq
import numpy as np
import PostImageData
import os
import base64


def findMaxAreaAndDraw(closed):
    contours, hierarchy = cv2.findContours(cv2.bitwise_not(closed[:, :, 0]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(genrateMaskWithHair, contours, -1, (0,255,0),1)
    mask_optimized_hair_face = np.zeros(closed.shape, np.uint8)
    max_contours = max(contours, key=cv2.contourArea)
    contours_list = []
    contours_list.append(max_contours)
    cv2.drawContours(mask_optimized_hair_face, contours_list, -1, (255, 255, 255), -1)
    closed = cv2.bitwise_not(mask_optimized_hair_face)
    # closed = cv2.merge([closed,closed,closed])
    return closed


def GetEyeArea(begin, end, i, faceOut, fa_wrap_begin_point):
    x_value_min = faceOut[i * 3 + 1][fa_wrap_begin_point + 2 * begin]
    for x in range(begin, end):
        # From 51 to 61
        if x_value_min > faceOut[i * 3 + 1][fa_wrap_begin_point + 2 * x]:
            x_value_min = faceOut[i * 3 + 1][fa_wrap_begin_point + 2 * x]
    return x_value_min


def memory_solve(face_list_source):
    try:
        face_list = []
        face_num = len(face_list_source)
        for i in range(0, face_num):
            face_list.append(face_list_source[i][0])
            face_list.append(face_list_source[i][1])
            face_list.append(face_list_source[i][2])
            face_list.append(face_list_source[i][3])
        del face_list_source
        return face_list, face_num
    except Exception as error:
        # print 'cannot get the face'
        print 'this is not the needed file'


def test_Hair_Segmentation_http_call_single(filename, filename_, UPLOAD_FOLDER_MASK):
    headers = {"Content-Type": "application/x-www-form-urlencoded",
               "Connection": "Keep-Alive"};
    #  define the header
    ip = "172.16.3.32"
    url = "/seg_hair_upload_json"
    socketPort = 1052
    name = filename_.split('.jpg')[0]
    result = PostImageData.http_call_url(filename_, filename, ip="172.16.3.32", url="/seg_hair_upload_json",
                                         wSelect='true', socketPort=socketPort)
    evalResult = eval(result)
    if evalResult.has_key("imageData"):  # get image data from json
        imgData = base64.b64decode(evalResult['imageData'])
        # leniyimg = open(os.path.join(dstImageFloder,str(index)+'_imgout.png'),'wb')
        leniyimg = open(os.path.join(UPLOAD_FOLDER_MASK, name + 'mask.png'), 'wb')
        leniyimg.write(imgData)
        leniyimg.close()
    return UPLOAD_FOLDER_MASK + filename_


def least_line_method_unlinear_x7(rightLine, Xi):
    def func(p, x):
        a, b, c, d, e, f, g = p
        return a * x * x * x * x * x * x + b * x * x * x * x * x + c * x * x * x * x + d * x * x * x + e * x * x + f * x + g
        # need to revise here if we change the dimension

    def error(p, x, y, s):
        print s
        return func(p, x) - y

    Xi = np.array(Xi, dtype=np.int64)
    Yi = np.array(rightLine, dtype=np.float64)
    # TEST
    p0 = [5, 2, 10, 1, 1, 1, 1]
    s = "Test the number of iteration"
    Para = leastsq(error, p0, args=(Xi, Yi, s))
    a, b, c, d, e, f, g = Para[0]
    return a, b, c, d, e, f, g, Xi, Yi


def run_single_image(src_path, pathname, upload_face_path, upload_mask_path):
    """
    :param src_path:  the whole input src image name
    :param pathname:  the src image name
    :param upload_face_path: the folder which saved the face
    :param upload_mask_path: the folder which saved the mask

    :return: face / face mask and error message
    """

    display_judgement = []
    display_judgement.append(True)
    display_judgement.append('Neural Photo Transform Success!')
    img_src = cv2.imread(src_path)

    # @Parameter
    FaceSize = 512
    #afExtentParam = [0.13, 0.0, 0.0, 0.0, 0.15]
    afExtentParam = [1.0, 0.0, 0.6, 0.0, 0.0]
    close_kernel_size = 100
    erode_kernel_size = 20
    hair_threshold = 255 - 180

    FDScore = 0
    FDScale = 0

    BorderMode = MT2Python.MT_BORDER_CONSTANT
    face_list, fd_t0, fa_t1, warp_t2 = MT2Python.FDFA.DetectWarpedFaceImages_s(img_src, afExtentParam, FaceSize,
                                                                               FDScore, nBorderMode=BorderMode)
    face_list, _ = memory_solve(face_list)
    tran_mat = face_list[0 * 4 + 2][18:26]
    # draw image here @yangming
    # face_show = draw_points_funtion(face_list, face_list[1])
    face_path = upload_face_path + pathname
    cv2.imwrite(face_path, face_list[0])
    mask = test_Hair_Segmentation_http_call_single(face_path, pathname, upload_mask_path)
    img = copy.deepcopy(face_list[0])
    img2 = face_list[0]
    rightLine_X = []
    rightLine_Y = []
    leftLine_Y = []
    leftLine_X = []
    wholeLine_X = []
    wholeLine_Y = []

    # face_list = NPE_face_list
    # fa_wrap_begin_point = 212
    try:
        for i in range(0, 30, 2):
            leftLine_X.append(face_list[1][i])  # left -> right
            leftLine_Y.append(face_list[1][i + 1])
        for i in range(32, 62, 2):
            rightLine_X.append(face_list[1][i])  # left -> right
            rightLine_Y.append(face_list[1][i + 1])
        for i in range(0 + 212, 62 + 212, 2):
            wholeLine_X.append(face_list[1][i])
            wholeLine_Y.append(face_list[1][i + 1])
            # Get the left
        a, b, c, d, e, f, g, Xi, Yi = least_line_method_unlinear_x7(leftLine_X, leftLine_Y)
        method_for_draw = 'least_line_method_unlinear'
        x = Xi
        x_newindex = []
        # for i in range(max(x)-50,max(x),1):
        for i in range(0, FaceSize - 1, 1):
            x_newindex.append(i)
        x_newindex = np.array(x_newindex, dtype=np.int64)

        x_right_control = max(x)
        x_begin_control = max(x) - 50

        x = x_newindex
        Yi = a * x * x * x * x * x * x + b * x * x * x * x * x + c * x * x * x * x + d * x * x * x + e * x * x + f * x + g
        final_leftLine = Yi
        a, b, c, d, e, f, g, Xi, Yi = least_line_method_unlinear_x7(rightLine_X, rightLine_Y)
        method_for_draw = 'least_line_method_unlinear'
        x = Xi
        x_newindex = []
        for i in range(0, FaceSize - 1, 1):
            x_newindex.append(i)
        x_newindex = np.array(x_newindex, dtype=np.int64)
        x = x_newindex
        Yi = a * x * x * x * x * x * x + b * x * x * x * x * x + c * x * x * x * x + d * x * x * x + e * x * x + f * x + g
        final_rightLine = Yi
        begin_line = []
        end_line = []
        img2[0:FaceSize] = 255
        for h in range(0, len(final_leftLine)):
            begin = int(final_leftLine[h])
            end = int(final_rightLine[h])
            if begin <= 0 or end <= 0 or begin >= 255 or end >= 255:
                print('minus value or out of index, jump')
            else:
                for z in range(begin, end):  # (begin/end) => (end/begin)
                    img2[h][z] = 0
                    begin_line.append(begin)
                    end_line.append(end)

        a, b, c, d, e, f, g, Xi, Yi = least_line_method_unlinear_x7(wholeLine_Y, wholeLine_X)
        method_for_draw = 'least_line_method_unlinear'
        # draw_line(a, b, c, d, e,Xi, Yi,method_for_draw)
        x = Xi
        x_newindex = []
        # for i in range(x_right_control,x_right_control+50,1):
        for i in range(0, FaceSize - 1, 1):
            x_newindex.append(i)
        x_newindex = np.array(x_newindex, dtype=np.int64)
        x = x_newindex
        Yi = a * x * x * x * x * x * x + b * x * x * x * x * x + c * x * x * x * x + d * x * x * x + e * x * x + f * x + g
        final_wholeLine = Yi
        img2[0:FaceSize] = 255
        for h in range(0, len(x)):
            begin = int(final_wholeLine[h])
            if begin <= 0:
                begin = 0
                print('line value minus value or out of index, set to zero')
            elif end >= 255:
                end = 255
                print('minus value or out of index, set to zero')
            try:
                # get the begin points and the end points, which calculated by the index.
                for z in range(0, begin):  # (begin/end) => (end/begin)
                    img2[z][h] = 0
                    begin_line.append(begin)
                    end_line.append(end)
            except:
                print('the face is out of range')
        # get the mask
        mask_path = mask.split('.jpg')[0] + 'mask.png'
        img_hair_source_mask = cv2.imread(mask_path)
        img_hair_source = cv2.resize(img_hair_source_mask, (FaceSize, FaceSize), interpolation=cv2.INTER_CUBIC)
        retval, img_hair = cv2.threshold(img_hair_source, hair_threshold, 255, cv2.THRESH_BINARY)

        # new mask revised here @yangming
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
        img2 = cv2.erode(img2, kernel)

        img3 = cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(img2), img_hair))

        # optimized operation
        genrateMaskWithHair_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(cv2.bitwise_not(genrateMaskWithHair_gray), cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(genrateMaskWithHair, contours, -1, (0,255,0),1)
        mask_optimized = np.zeros(img3.shape, np.uint8)
        max_contours = max(contours, key=cv2.contourArea)
        contours_list = []
        contours_list.append(max_contours)
        cv2.drawContours(mask_optimized, contours_list, -1, (255, 255, 255), -1)
        img3 = cv2.bitwise_not(mask_optimized)

        # hat optimized and hair optimized
        mask_min_x_left = GetEyeArea(33, 41, 0, face_list, 212)
        mask_min_x_right = GetEyeArea(42, 51, 0, face_list, 212)

        final_clear_index_x = np.where(img_hair[:, :, 0] == 0)[0].min()
        min_fa_points = 0
        if mask_min_x_left > mask_min_x_right:
            min_fa_points = mask_min_x_right
        else:
            min_fa_points = mask_min_x_left

        if final_clear_index_x > int(min_fa_points):
            final_clear_index_x = int(min_fa_points)

        img3[0:final_clear_index_x, :, :] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        closed = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel)

        # if hair_add_choice == 'true':
        #     closed = cv2.bitwise_and(closed, img_hair)
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        #     closed = findMaxAreaAndDraw(cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel))
        # img2_bg_closed_dilated = cv2.bitwise_and(img, cv2.bitwise_not(closed))

        width = img_src.shape[0]
        height = img_src.shape[1]
        #  img2_bg_closed_dilated =
        # MT2Python.FDFA.InverseWarpedFaceImages(cv2.bitwise_not(img2_bg_closed_dilated),
        # img2_bg_closed_dilated[:,:,0], tran_mat, height, width, 1, 1)[0]
        closed = MT2Python.FDFA.InverseWarpedFaceImages(cv2.bitwise_not(closed), closed[:,:,0], tran_mat, height, width, 1, 1)[0]
        return closed, display_judgement
    except:
        display_judgement[0] = False
        display_judgement[1] = 'Cannot detect face on this image, please change the uploaded image :)'
        print('Cannot find the fa points')
        NPE_save_path = ""
        img2_bg_closed_dilated = ""
        return closed, display_judgement


if __name__ == '__main__':
    # parameters
    src_path = '/home/wenyangming/DLwebdemo/a11.jpg'
    pathname = 'a11.jpg'
    upload_mask_path = '/home/wenyangming/DLwebdemo/MaskTest/'
    upload_face_path = '/home/wenyangming/DLwebdemo/FaceTest/'

    # get the hair mask
    # mask_path = test_Hair_Segmentation_http_call_single(src_path, pathname, upload_mask_path)

    # get the face mask
    face_mask, _ = run_single_image(src_path, pathname, upload_face_path, upload_mask_path)
    cv2.imwrite(upload_face_path + pathname, face_mask)
