# -*- coding: utf-8 -*-
import cv2
import os

#動画ファイルを読み込む
file_name = "/home/yamashita/private/takebe/media5.avi"
video = cv2.VideoCapture(file_name)
#スクリーンキャプチャを保存するディレクトリ
dir_name = "screen_caps5"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
#フレーム数を取得
frame_count = int(video.get(7))
for i in range(frame_count):
    _, frame = video.read()
    cv2.imwrite(dir_name+ "/" + str(i) + ".png", frame)
