import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results


'''
進行指標評估需要注意以下幾點：
1、該文件生成的圖為灰度圖，因為值比較小，按照JPG形式的圖看是沒有顯示效果的，所以看到近似全黑的圖是正常的。
2、該文件計算的是驗證集的miou，當前該庫將測試集當作驗證集使用，不單獨劃分測試集
3、僅有按照VOC格式數據訓練的模型可以利用這個文件進行miou的計算。
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用於指定該文件運行時計算的內容
    #   miou_mode為0代表整個miou計算流程，包括獲得預測結果、計算miou。
    #   miou_mode為1代表僅僅獲得預測結果。
    #   miou_mode為2代表僅僅計算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分類個數+1、如2+1
    #------------------------------#
    num_classes     = 21
    #--------------------------------------------#
    #   區分的種類，和json_to_dataset裡面的一樣
    #--------------------------------------------#
    name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC數據集所在的文件夾
    #   默認指向根目錄下的VOC數據集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
