import numpy as np
import os
import cv2
import Functions_Modules

imgName = 'Sick_1_C1P12E2'
numberOfColors = 3
windowSize = 5

Imagem = cv2.imread(f'Sampled Images/{imgName}.jpg')
Imagem, MP_Classes = Functions_Modules.QUANTIFICAR(Imagem, numberOfColors)
cv2.imwrite(f'examples/{imgName}_quantificada.png', Imagem)
#cv2.imwrite(f'examples/{imgName}_quantificada_{numberOfColors}.png', Imagem)
jImage = Functions_Modules.J_IMAGE(MP_Classes, windowSize)

jImageGreyScale = Functions_Modules.J_IMAGE_TO_GREYSCALE(jImage)

cv2.imwrite(
    f'examples/{imgName}_j.jpg', jImageGreyScale)
#cv2.imwrite(f'examples/{imgName}_j_c{numberOfColors}_w{windowSize}.jpg', jImageGreyScale)
