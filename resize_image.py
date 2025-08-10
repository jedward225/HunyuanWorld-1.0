from PIL import Image
img = Image.open('/home/liujiajun/HunyuanWorld-1.0/test_results/street/street.jpg') 
target_size = (1920, 960)
img = img.resize(target_size, Image.LANCZOS) 
img.save('/home/liujiajun/HunyuanWorld-1.0/test_results/street/panorama.png') 