from PIL import Image
filename = input()

img = Image.open(filename)
(width, height) = (img.width//6, img.height//6)
img_resized = img.resize((width, height))
img_resized.save('in3.jpeg', quality = 100)
