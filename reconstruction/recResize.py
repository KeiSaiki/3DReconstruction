import os
import pyheif
from PIL import Image
import initialSetup as istu

RESIZE_FACTOR = 6

def convert_heic_to_jpeg(heic_file, jpeg_file):
    heif_file = pyheif.read(heic_file)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    image.save(jpeg_file, "JPEG")

# 指定されたディレクトリ内の画像のサイズを変更する関数
def resize_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(directory, filename)
            try:
                img = Image.open(img_path)
                new_size = (img.width // RESIZE_FACTOR, img.height // RESIZE_FACTOR)
                img_resized = img.resize(new_size)
                img_resized.save(os.path.join(directory, f"{os.path.splitext(filename)[0]}_resized{os.path.splitext(filename)[1]}"))
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# 'initialSetup.py' からディレクトリ名を取得する
directory = istu.original_image_dir

# ディレクトリ名が正しく取得できなかった場合、ユーザーに再入力を求める
while not directory:
    print("Could not retrieve directory name from 'initialSetup.py'. Please enter the directory path:")
    directory = input()

# 画像のサイズを6で割る
resize_images_in_directory(directory)

print("Image resizing completed.")
