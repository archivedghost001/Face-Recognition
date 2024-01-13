# Penjelasan kode tentang proyek face detection
F-string merupakan fitur pada Python yang memungkinakan kita untuk memasukkan ekspresi atau variable ke dalam string dengan cara yang lebih mudah dan intuitif

```python
# Combine the directory output and the name file with the ending "_detected"
output_image_path = os.path.join(
    output_directory, f"{name}_detected{extension}")
```
Dalam contoh ini, f-string digunakan untuk menggabungkan nilai variabel name, "_detected", dan nilai variabel extension menjadi satu string. Dengan menggunakan f-string, kita dapat langsung memasukkan ekspresi atau variabel dalam kurung kurawal {} ke dalam string tanpa perlu melakukan operasi penggabungan manual atau konversi tipe data.

Sebagai contoh, jika nilai dari variabel name adalah "gambar" dan nilai dari variabel extension adalah ".jpg", maka hasil dari ekspresi tersebut akan menjadi "gambar_detected.jpg".
