#### Paramaters description
```
# Function: Face Search 
# Author: wym@meitu.com
# Main develop environments: develop branch 
# @main function: FaceMaskMultprocess.py
#--------------------------------------------------------------------------------

# @For data team 

#参数为batch/filterImage:
#parser.add_argument('--batch_image_folder', type=str, default='/home/wenyangming/meituphone/20170410/XinNiangImage2') 
#原始图片即直接从网上下下来的图片的目录
#parser.add_argument('--function', type=str, default='filterImage') 
#会在上级目录生成4个文件夹,得出脸上画框框即第一次filter结果

#参数为re_filter:
#if args.single_batch_choice == 're_filter'                        
#得出最后需要的结果

# @path setting
#src_path = 'D:/20170410XinNiangImage2/XinNiangImage2/'           
#原始图片即直接从网上下下来的图片的目录
#dst_filtered_path = 'D:/20170413BeautifulGirls/'                 
#外包清理后四个文件夹的目录
#会在src path下生成4个文件夹,得到最终裁脸的结果

#--------------------------------------------------------------------------------

# @For developer

# @param : filter the threshold of resolution
# means only maintain 256x256
    threshold = 256
    
# @param : folder
    NormalSrcImageFolder = "/home/wenyangming/facecrop/TestImage/Normal/"
    NormalDstImageFolder = "/home/wenyangming/facecrop/TestImage/Normal_out/"
    BeautySrcImageFolder = "/home/wenyangming/facecrop/TestImage/Beauty/"
    BeautyDstImageFolder = "/home/wenyangming/facecrop/TestImage/Beauty_out/"
    targetPointsFile = "/home/wenyangming/facecrop/TestImage/BeautyPointsList.txt"
    OutPutImageFolder = "/home/wenyangming/facecrop/TestImage/result/"
    
# @function : main interface here for getting the crop-face image and save to dstFolder
    cropFaceAndWrite(NormalSrcImageFolder, NormalDstImageFolder)
    cropFaceAndWrite(BeautySrcImageFolder, BeautyDstImageFolder)

# @function : get the shortest image and concat
    CalculateDistanceAndSave(BeautyDstImageFolder, NormalDstImageFolder, targetPointsFile, OutPutImageFolder)
    
# @param [afExtentParam] input: 人脸框内框向外延拓的参数, 延拓参数从头部，逆时针存储

#  afExtentParam[0]   沿额头方向延拓的参数
#  afExtentParam[1]   沿脸庞左边延拓的参数
#  afExtentParam[2]    沿下巴方向延拓的参数
#  afExtentParam[3]  沿脸庞右边延拓的参数
#  afExtentParam[4] : 在上述延拓的基础上，4个方向统一延拓的参数, 即长宽放大的倍数为 1+fExtentRect

# FaceSize: 裁剪人脸图片的大小，内部会缩放到该大小
# FDScore: 人脸检测的分数，0-160
# return face_list[0] 裁剪的人脸图像， face_list[1] FA得到的人脸点， face_list[2] FD得到的人脸矩形框， 
# return face_list[3] Pose Estimation得到的人脸点, 点数为6, 姿态的旋转角度(3 x 1),姿态的位移向量(3 x 1)
# 依次类推，每4个代表一个人脸

```

