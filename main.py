import numpy as np
import os
import cv2
import Functions_Modules

Imagem = cv2.imread('Sampled Images\Sick_1_C5P9H2.jpg')
Imagem, MP_Classes = Functions_Modules.QUANTIFICAR(Imagem, 5)
out = Functions_Modules.J_IMAGE(Imagem, MP_Classes)
for x in out:
    print(list(x))
