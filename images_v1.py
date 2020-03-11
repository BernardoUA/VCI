import numpy as np
import cv2 as cv

img = cv.imread('image.jpg',4) 
# Lê uma imagem a preto e branco (cinzento) e guarda-a na variável img

cv.imshow('image',img)
# Mostra uma imagem numa janela

k = cv.waitKey(0) 
# waitKey é uma função que espera por uma atividade no teclado. 
# 0 quer dizer que ele simplesmente vai esperar por qualquer atividade
# no teclado 

if k == 27:         # Espera pela tecla ESC para sair
    cv.destroyAllWindows()
    # Fecha todas as janelas abertas
    
elif k == ord('s'): # Espera pela tecla 's' para guardar e sair
    cv.imwrite('image_gray.png',img)
    # Guarda a imagem de img, com o nome image_gray.png
    cv.destroyAllWindows()
    # Fecha todas as janelas abertas
    
