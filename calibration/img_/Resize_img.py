# ライブラリをインポート
import cv2
import glob

# フォルダ内の画像を取得
images = glob.glob('*.jpeg')

for filename in images:

    # 指定したパスの画像を読込む
    img = cv2.imread(filename)

    # 高さと幅を定義する
    height = img.shape[0]
    width = img.shape[1]

    # 画像をリサイズする。今回は各長さを6分の1にする。800万画素➔22万画素に縮小
    img_resize = cv2.resize(img, (int(width/6), int(height/6)))

    # リサイズ後の画像を保存する
    cv2.imwrite('./img_resized/' + str(filename), img_resize)