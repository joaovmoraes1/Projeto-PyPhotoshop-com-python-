from PIL import Image, ImageFilter

# Carregue a imagem
img = Image.open('input/city.png')
print('Imagem carregada:', img)

# Aplique filtros
img_bright = img.point(lambda p: p * 1.2)
print('Filtro de brilho aplicado.')
img_contrast = img_bright.point(lambda p: p * 1.5 + 0.5)
print('Filtro de contraste aplicado.')

img_contrast = img_contrast.convert('L')  # Converta para escala de cinza
img_blur = img_contrast.filter(ImageFilter.GaussianBlur(radius=2.5))
print('Filtro Gaussian Blur aplicado.')
img_edges = img_blur.filter(ImageFilter.FIND_EDGES)
print('Filtro de detecção de bordas aplicado.')

# Salve a imagem resultante
img_edges.save('output/output_image.png')
print('Imagem salva em:', 'output/output_image.png')