import numpy as np
import os
import cv2


# Soma de valores de cor
def soma_Distancias(tupleA, tupleB):
    input = [tupleA, tupleB]
    ouput = []
    for ax1 in input:
        temp = []
        for ax2 in ax1:
            temp.append(float(ax2))
        ouput.append(temp)
    return [ouput[0][0] + ouput[1][0], (ouput[0][1] - ouput[1][1]), (ouput[0][2] - ouput[1][2])]


# Subtração de valores de cor para calcular a distancia
def distancia_tridimensional(tupleA, tupleB):
    input = [tupleA, tupleB]
    ouput = []
    for ax1 in input:
        temp = []
        for ax2 in ax1:
            temp.append(int(ax2))
        ouput.append(temp)
    soma = ((ouput[0][0]-ouput[1][0])**2) +((ouput[0][1]-ouput[1][1])**2) +((ouput[0][2]-ouput[1][2])**2)
    return np.sqrt(soma)


# Função para quantificar imagens coloridas e retornar Mapa de classes
def QUANTIFICAR(image, k):
    #   Trasforma a imagem em formato viavel para kmeans
    i = np.float32(image).reshape(-1, 3)

    #   Seleciona os ponto medio de k intervalos de pixels na imagem
    ret, label, center = cv2.kmeans(i, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

    #   Trasforma saida center em um tipo apropiado para imagens
    center = np.uint8(center)

    #   Cria uma iamgem utilizando a lista de valores medios aplicando MP_Classes de classe como indice
    qnt_Image = center[label.flatten()]
    qnt_Image = qnt_Image.reshape(image.shape)

    # Cria matris que mostra as classes de cada pixels na imnagem
    cls_Map = np.reshape(label, (qnt_Image.shape[0], qnt_Image.shape[1]))

    return qnt_Image, cls_Map


def SEGMENTAR(quantifiedImage,kernel_dim):
    gray = cv2.cvtColor(quantifiedImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    # sure background area
    sure_bg = cv2.dilate(thresh, kernel, 5)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(quantifiedImage, markers)
    quantifiedImage[markers == -1] = [255, 255, 255]

    return quantifiedImage


def J_IMAGE(Quantized_Image, ClassMap):
    #   Imagem J
    jImage = []
    #   Numero de Instancias de cada classe
    ArrayClassMean = []
    #   Array com de classes em ordem
    ArrayClass = list(ClassMap.flatten())
    #   Numero de cores no mapa
    k = len(np.unique(ArrayClass))
    #   Cor media na imagem
    mean = np.average(np.average(Quantized_Image, axis=0), axis=0)

    # Calcula a media dos de cada Classe
    for f in range(k):
        soma = [0, 0, 0]
        total = ArrayClass.count(f)
        for g in range(len(ClassMap)):
            for h in range(len(ClassMap[g])):
                if f == ClassMap[g][h]:
                    px_color = Quantized_Image[g][h]
                    soma = soma_Distancias(px_color, soma)
        ArrayClassMean.append([soma[0]/total, (soma[1])/total, soma[2]/total])

    #   Arrays com ST e SW para cada pixel
    for l in range(len(ClassMap)):
        for m in range(len(ClassMap[l])):
            # Imagem ST para cada pixel
            ST_Image = []
            #   Imagem SW para cada pixel
            SW_Image = []
            for z in range(len(ClassMap)):
                for y in range(len(ClassMap[z])):
                    ST_Image.append(distancia_tridimensional(Quantized_Image[z][y], mean) ** 2)

                    SW_Image.append(distancia_tridimensional(Quantized_Image[z][y], ArrayClassMean[ClassMap[l][m]]) ** 2)

            ST = sum(ST_Image)
            jImage.append(0)

    return np.array(jImage).reshape(ClassMap.shape)
