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
    soma = ((ouput[0][0]-ouput[1][0])**2) + ((ouput[0][1] -
                                              ouput[1][1])**2) + ((ouput[0][2]-ouput[1][2])**2)
    return np.sqrt(soma)


# Função para quantificar imagens coloridas e retornar Mapa de classes
def QUANTIFICAR(image, k):
    #   Trasforma a imagem em formato viavel para kmeans
    i = np.float32(image).reshape(-1, 3)

    #   Seleciona os ponto medio de k intervalos de pixels na imagem
    ret, label, center = cv2.kmeans(i, k, None, (cv2.TERM_CRITERIA_EPS +
                                    cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

    #   Trasforma saida center em um tipo apropiado para imagens
    center = np.uint8(center)

    #   Cria uma iamgem utilizando a lista de valores medios aplicando MP_Classes de classe como indice
    qnt_Image = center[label.flatten()]
    qnt_Image = qnt_Image.reshape(image.shape)

    # Cria matris que mostra as classes de cada pixels na imnagem
    cls_Map = np.reshape(label, (qnt_Image.shape[0], qnt_Image.shape[1]))

    return qnt_Image, cls_Map


def SEGMENTAR(quantifiedImage, kernel_dim):
    gray = cv2.cvtColor(quantifiedImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    # sure background area
    sure_bg = cv2.dilate(thresh, kernel, 5)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.4 * dist_transform.max(), 255, 0)
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


def euclidianDistance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calcMeanPosition(points):
    x = 0
    y = 0
    for point in points:
        x += point[0]
        y += point[1]
    return [x / len(points), y / len(points)]


def calcSt(points, mean_point):
    st = 0
    for point in points:
        st += euclidianDistance(point, mean_point) ** 2
    return st


def calcSw(classMap):
    sw = 0
    classToPixelsMap = calcClassToPixelsMap(classMap)
    for colorClass, pixels in classToPixelsMap.items():
        mean = calcMeanPosition(pixels)
        sw += calcSt(pixels, mean)
    return sw


def calcClassToPixelsMap(classMap):
    classToPixelsMap = {}
    for row_idx in range(len(classMap)):
        for col_idx in range(len(classMap[row_idx])):
            colorClass = classMap[row_idx][col_idx]
            list_of_pixels = classToPixelsMap[colorClass] = classToPixelsMap.get(
                colorClass, [])
            list_of_pixels.append([row_idx, col_idx])
    return classToPixelsMap


def calcListOfPixelsPositions(img):
    listOfPixels = []
    for row_idx in range(len(img)):
        for col_idx in range(len(img[row_idx])):
            listOfPixels.append([row_idx, col_idx])
    return listOfPixels


def calcClassToColorMap(quantizedImage, classMap):
    classToColorMap = {}
    for row_idx in range(len(classMap)):
        for col_idx in range(len(classMap[row_idx])):
            classToColorMap[classMap[row_idx][col_idx]
                            ] = quantizedImage[row_idx][col_idx]
    return classToColorMap


def calcClassToPixelQuantityMap(classMap):
    classToPixelQuantityMap = {}
    for row_idx in range(len(classMap)):
        for col_idx in range(len(classMap[row_idx])):
            classToPixelQuantityMap[classMap[row_idx][col_idx]] = classToPixelQuantityMap.get(
                classMap[row_idx][col_idx], 0) + 1
    return classToPixelQuantityMap


def centerCoordinates(img):
    height, width, _ = img.shape
    return (int(width/2), int(height/2))


def getWindow(img, x, y, width, height):
    for x_offset in range(width):
        for y_offset in range(height):
            yield img[x + x_offset][y + y_offset]


def J_IMAGE(Quantized_Image, ClassMap):

    st = calcSt(ClassMap, centerCoordinates(Quantized_Image))
    sw = calcSw(ClassMap)

    j = (st - sw)/sw

    print('st', st)
    print('sw', sw)
    print('j', j)

    exit()
