import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'ch2-ps431-jelajahi-credentials.json'
storage_client = storage.Client()

def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req

model = load_model('batik_model.h5', custom_objects={'req': req})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            image_bucket = storage_client.get_bucket(
                'jelajahi')
            filename = request.json['filename']
            img_blob = image_bucket.blob('predict/' + filename)
            img_path = BytesIO(img_blob.download_as_bytes())
        except Exception:
            respond = jsonify({'message': 'Error loading image file'})
            respond.status_code = 400
            return respond

        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        images = np.vstack([img_array])

        # model predict
        batik_predict = model.predict(images)

        nama = ['Batik Lasem', 'Batik Parang', 'Batik Pati', 'Batik Pekalongan', 'Batik Sekar Jagad',
                'Batik Sidoluhur', 'Batik Sogan', 'Batik Truntum']
        tipe = ['Batik', 'Batik', 'Batik', 'Batik', 'Batik', 'Batik', 'Batik', 'Batik']
        origin = ['Rembang', 'Solo', 'Pati', 'Pekalongan', 'Solo', 'Banyumas', 'Solo', 'Solo']
        Latitude = ['-6.707107734936229', '-7.567375126550622', '-6.753229640231162', '-6.892020565996946', 
                    '-7.567375126550622', '-7.567375126550622', '-7.567375126550622', '-7.567375126550622']
        Longitude = ['111.3318315373741', '110.82859560170351', '111.0360452054126', '109.68239872415117', 
                     '110.82859560170351', '109.27606211053724', '110.82859560170351', '110.82859560170351']
        province = ['Jawa Tengah', 'Jawa Tengah', 'Jawa Tengah', 'Jawa Tengah', 'Jawa Tengah', 'Jawa Tengah', 'Jawa Tengah', 'Jawa Tengah']
        deskripsi = ['Batik Lasem terkenal dengan warna-warna kontras dan motif yang kompleks. Motifnya sering kali terinspirasi oleh flora dan fauna.', 
                     'Batik Parang dikenal dengan motif geometris yang menyerupai bentuk parang (pedang). Motif ini sering diartikan sebagai simbol keberanian dan kekuatan.', 
                     'Batik Pati memiliki motif-motif yang bervariasi, mencakup pola-pola geometris dan floral. Warna-warna yang digunakan bisa menciptakan tampilan yang elegan dan menarik.', 
                     'Batik Pekalongan memiliki keunikan dalam penggunaan warna-warna cerah dan campuran motif-motif dari berbagai budaya, membuatnya terlihat eksotis dan artistik.', 
                     'Batik Sekar Jagad memiliki motif bunga-bunga yang indah dan beraneka ragam. Motif ini mencerminkan keindahan alam dan kehidupan.', 
                     'Batik Sidoluhur memiliki motif tradisional yang menggambarkan kehidupan sehari-hari di daerah Banyumas, Jawa Tengah.', 
                     'Batik Sogan memiliki warna dasar yang cenderung kecokelatan dan motif-motif yang khas. Motif ini sering kali terinspirasi oleh kearifan lokal.', 
                     'Batik Truntum memiliki motif utama berupa pola jalinan yang melambangkan ikatan cinta, persatuan, dan kesetiaan. Motif ini sering diartikan sebagai simbol keharmonisan dalam hubungan, baik dalam pernikahan maupun persahabatan. Selain itu, batik ini juga menampilkan berbagai motif geometris dan floral yang melengkapi kesan elegan dan indah dari Batik Truntum. Warna-warna yang digunakan cenderung cerah dan kontras, menambah kecantikan dari setiap karya Batik Truntum.']

        result = {
            "nama": nama[np.argmax(batik_predict)],
            "tipe": tipe[np.argmax(batik_predict)],
            "province": province[np.argmax(batik_predict)],
            "deskripsi": deskripsi[np.argmax(batik_predict)],
            "origin": origin[np.argmax(batik_predict)],
            "latitude": Latitude[np.argmax(batik_predict)],
            "longitude": Longitude[np.argmax(batik_predict)]
        }

        respond = jsonify(result)
        respond.status_code = 200
        return respond

    return 'Silahkan masukkan gambar'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8000')