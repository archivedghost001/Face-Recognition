import numpy as np
import cv2

# Jalur prototxt model Caffe
prototxt_path = "model-data/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "model-data/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# menginsialisasi pengenalan wajah (default face haarcascade)
face_cascade = cv2.CascadeClassifier("model-data/face.xml")

# memuat model caffe data
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# membuat objek kamera baru
capture = cv2.VideoCapture(0)

while True:
    # membaca gambar/fram dari kamera
    live, camera = capture.read()

    # mendapatkan lebar dan tinggi gambar
    h, w = camera.shape[:2]

    # menkonversi gambar ke skala abu2
    image_gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)

    # mendeteksi semua wajah pada kamera
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # menghitung jumlah wajah yang terdeteksi
    face_count = len(faces)

    # pra-pemrosesan gambar: resize dan pengurangan mean (rata-rata)
    blob = cv2.dnn.blobFromImage(
        camera, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # menetapkan gambar menjadi input jaringan saraf
    model.setInput(blob)

    # melakukan inferensi dan mendapatkan hasilnya
    output = np.squeeze(model.forward())

    # mengatur ukuran font dan font style
    font_scale = 1
    font_style = cv2.FONT_HERSHEY_SIMPLEX

    # buat persegi panjang untuk mendeteksi wajah dengan perulangan
    for i in range(0, output.shape[0]):
        # membuat variabel facialAccuracy untuk output looping i
        facialAccuracy = output[i, 2]

        # jika akurasi wajah di atas 50%, maka gambarkan kotak sekitarnya
        if facialAccuracy > 0.5:
            # get koordinat kotak sekitarnya dan memperbesar ukurannya ke gambar asli
            box = output[i, 3:7]*np.array([w, h, w, h])
            # mengkonversi ke integer
            start_x, start_y, end_x, end_y = box.astype(np.int64)
            # menggambar persegi panjang di sekitar wajah
            cv2.rectangle(
                camera,
                (start_x, start_y),
                (end_x, end_y),
                color=(0, 0, 255),
                thickness=4,
            )
            # membuat teksnya juga diatas persegi panjang
            cv2.putText(
                camera,
                f"Warcriminal {facialAccuracy*100:.2f}%",
                (start_x, start_y - 5),
                font_style,
                font_scale,
                (0, 0, 255),
                2
            )
            # mencetak jumlah wajah yang terdeteksi
            if face_count > 1:
                print(f"{face_count} warcriminals detected", end="\r")
            else:
                print(f"{face_count} warcriminal detected", end="\r")

    # menampilkan jendela baru
    cv2.imshow("Ry-tech Sys v5.3", camera)

    # jika pengguna menekan tombol "q" maka loop akan berhenti
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
