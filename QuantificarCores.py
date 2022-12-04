import numpy as np
import os
import cv2


# Função para quantificar imagens coloridas

def clr_Quant(image, k):
    i = np.float32(image).reshape(-1, 3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(i, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    qnt_Image = center[label.flatten()]
    qnt_Image = qnt_Image.reshape(image.shape)
    return qnt_Image

# Lista de arquivos na pasta
samples = os.listdir("Sampled Images")

# Opera todos os arquivos na pasta
for path in samples:
    etapaA = cv2.imread(f"Sampled Images/{path}")
    # cv2.imshow(path, etapaA)
    # cv2.waitKey(10)

    k = 15
    output = clr_Quant(etapaA, k)

    # cv2.imshow(path, output)
    # cv2.waitKey(10)


    # Salvar imagens
    cv2.imwrite(f"Quantified Photos/K{k}_{path}", output)