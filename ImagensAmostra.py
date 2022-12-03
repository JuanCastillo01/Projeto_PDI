import numpy as np
import os
import cv2
import random

i = 1
ImagesIndex = random.sample(range(1560), 50)

ListaDeFotos = os.listdir("Photos")

for IMG_Path_Index in ImagesIndex:
    print(i)
    example = cv2.imread(f"Photos/{ListaDeFotos[IMG_Path_Index]}")

    cv2.imshow("", example)
    cv2.waitKey(100)

    Final_Size_X = 380
    Final_Size_Y = 640

    CurrentX = example.shape[0]
    CurrentY = example.shape[1]

    DimX_Start = int((CurrentX/2) - (Final_Size_X/2))
    DimX_End = int((CurrentX/2) + (Final_Size_X/2))

    DimY_Start = int((CurrentY/2) - (Final_Size_Y/2))
    DimY_End = int((CurrentY/2) + (Final_Size_Y/2))

    example2 = example[DimX_Start:DimX_End, DimY_Start:DimY_End]

    cv2.imshow("", example2)
    cv2.waitKey(100)
    cv2.imwrite(f"Sampled Images/{ListaDeFotos[IMG_Path_Index]}", example2)
    i = i + 1

    if (i == 50):
        break

print("OK")