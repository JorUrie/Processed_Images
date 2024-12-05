# Importando bibliotecas
import pandas as pd
import numpy as np
import cv2
#-----------------------------------------------------------------------------------
def Image_MedianBlur(Name_Image):
    img = cv2.imread(Name_Image) # imread lee las imagenes con los pixeles codificados como enteros en el rango 0-255.
    # ------------------------------------------------------------------------------
    processed_image = cv2.medianBlur(img, 3)
    # save image to disk
    cv2.imwrite('processed_image.jpg', processed_image)
    # pause the execution of the script until a key on the keyboard is pressed
    cv2.waitKey(0)
    print('The processed image is ready!')
# -----------------------------------------------------------------------------------
def Image_GaussianBlur(Name_Image):
    img = cv2.imread(Name_Image) # imread lee las imagenes con los pixeles codificados como enteros en el rango 0-255.
    # ------------------------------------------------------------------------------
    processed_image = cv2.GaussianBlur(img,(5,5),0)
    # save image to disk
    cv2.imwrite('processed_image.jpg', processed_image)
    # pause the execution of the script until a key on the keyboard is pressed
    cv2.waitKey(0)
    print('The processed image is ready!')
#-----------------------------------------------------------------------------------
def Image_Blur(Name_Image):
    img = cv2.imread(Name_Image) # imread lee las imagenes con los pixeles codificados como enteros en el rango 0-255.
    # ------------------------------------------------------------------------------
    processed_image = cv2.blur(img, (5, 5))
    cv2.imwrite('processed_image.jpg', processed_image)
    # pause the execution of the script until a key on the keyboard is pressed
    cv2.waitKey(0)
    print('The processed image is ready!')
#-----------------------------------------------------------------------------------
def Image_BilateralFilter(Name_Image):
    img = cv2.imread(Name_Image) # imread lee las imagenes con los pixeles codificados como enteros en el rango 0-255.
    # ------------------------------------------------------------------------------
    processed_image = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite('processed_image.jpg', processed_image)
    # pause the execution of the script until a key on the keyboard is pressed
    cv2.waitKey(0)
    print('The processed image is ready!')
#-----------------------------------------------------------------------------------
def Menu():
    # Haciendo un menu para presentar los filtros de suavizado
    List_filter = {'Filter': ['medianBlur','GaussianBlur', 'blur', 'bilateralFilter'], 'Option': [1, 2, 3, 4]}
    df = pd.DataFrame(data = List_filter)
    print('Select one filter \n', df)
    opt = input(int())
    return int(opt)
#-----------------------------------------------------------------------------------
def selectFilter(Image):
    Option = Menu()
    print(Option)
    if Option == 1:
        Image_MedianBlur(Image)
    if Option == 2:
        Image_GaussianBlur(Image)
    if Option == 3:
        Image_Blur(Image)
    if Option == 4:
        Image_BilateralFilter(Image)
#-----------------------------------------------------------------------------------
def Image_porcessed_Fourier(Image):
    img = cv2.imread(Image)

    x = y = 600
    img = cv2.resize(img, (x, y))

    # GRISES
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = np.float64(gray)

    # TRANSFORMADA DE FOURIER EN 2D
    frr = np.fft.fft2(gray2)
    frr = np.fft.fftshift(frr)

    # CALCULAR LA MAGNITUD DEL ARREGLO
    frr_abs = np.abs(frr)

    # ESPECTRO DE FRECUENCIA EN ESCALA LOGARITMICA
    frr_log = 20 * np.log10(frr_abs)

    # MOSTRAMOS LA IMAGEN
    cv2.imshow("Imagen Original", img)
    img_frr = np.uint8(255 * frr_log / np.max(frr_log))
    #cv2.imshow("Espectro de Fourier Logaritmica", img_frr)

    # FILTRO PASA ALTO
    # Parte central valores cercanos al cero.
    # y el resto de valores sean altos
    F1 = np.arange(-x / 2 + 1, x / 2 + 1, 1)
    F2 = np.arange(-y / 2 + 1, y / 2 + 1, 1)
    [X, Y] = np.meshgrid(F1, F2)  # arreglo matricial de las combinaciones
    D = np.sqrt(X ** 2 + Y ** 2)  # distancia del centro
    D = D / np.max(D)
    # DEFINIR RADIO DE CORTE
    Do = 0.10
    # Creación del Filtro Ideal en 2D
    Huv = np.zeros((x, y))  # matriz de ceros
    # PRIMERO CREAR EL FILTRO PASA BAJO IDEAL
    for i in range(x):
        for j in range(y):
            if (D[i, j] < Do):
                Huv[i, j] = 1
    # CONVERTIR A PASA ALTO IDEAL
    Huv = 1 - Huv

    # ----------------------------------------------------
    #cv2.imshow("FILTRO 2D PASA ALTO IDEAL", np.uint8(255 * Huv))

    # --------------------------FILTRADO EN FRECUENCIA
    # -MULTIPLICACIÓN ELEMENTO A ELEMENTO EN EL DOMINIO DE LA FRECUENCIA
    Guv = Huv * frr
    # MAGNITUD
    Guv_abs = np.abs(Guv)
    Guv_abs = np.uint8(255 * Guv_abs / np.max(Guv_abs))
    #cv2.imshow('ESPECTRO DE FRECUENCIA G', Guv_abs)
    # ---TRANSFORMADA INVERSA PARA OBTENER LA SEÑAL FILTRADA
    # IFFT2
    gxy = np.fft.ifft2(Guv)
    gxy = np.abs(gxy)
    gxy = np.uint8(gxy)
    # --MOSTRAR LA IMAGEN FILTRADA
    cv2.imshow('IMAGEN FILTRADA', gxy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------

# Funciones ejecutadas
selectFilter("Photo.jpg")
