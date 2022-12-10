import numpy as np
import os
import cv2
import Functions_Modules

examples = [
    ['Sick_3_C9P23E1', 10, 4],
    ['Sick_2_C10P5E1', 6, 7],
    ['Sick_e_C5P24E2', 7, 8],
    ['Sick_1_C12P40E2', 10, 9],
]

for example in examples:
    imgName = example[0]
    print("Processing example: ", imgName)
    Imagem = cv2.imread(f'Sampled Images\{imgName}.jpg')
    Imagem, MP_Classes = Functions_Modules.QUANTIFICAR(Imagem, example[1])
    cv2.imwrite(f'examples/{imgName}_quantificada.jpg', Imagem)
    jImage = Functions_Modules.J_IMAGE(MP_Classes, example[2])

    jImageGreyScale = Functions_Modules.J_IMAGE_TO_GREYSCALE(jImage)

    cv2.imwrite(f'examples/{imgName}_jImage.png', jImageGreyScale)
