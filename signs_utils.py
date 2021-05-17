import shutil
import cv2
import numpy
import os

class Region:
    def __init__(self, img, coord):
        self.img = img
        self.coord = coord
        self.type = None
        self.score = 0


def load_training_data(train_path):
    """Carga las imágenes de entrenamiento devolviendolas agrupadas según su tipo.
        OUT: [tipo_señal: [img1, img2, ...], ...] """

    print('Cargando imágenes de entrenamiento...')

    prohib_dirs = ['00', '01', '02', '03', '04', '05', '07', '08', '09', '10', '15', '16']
    peligro_dirs = ['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    stop_dirs = ['14']

    prohib_imgs, peligro_imgs, stop_imgs = ([] for i in range(3))

    for dir in os.listdir(train_path):
        dir_path = os.path.join(train_path, dir)

        if dir in prohib_dirs:
            imgs, _ = __extract_imgs_in_dir(dir_path)
            prohib_imgs.extend(imgs)

        elif dir in peligro_dirs:
            imgs, _ = __extract_imgs_in_dir(dir_path)
            peligro_imgs.extend(imgs)

        elif dir in stop_dirs:
            imgs, _ = __extract_imgs_in_dir(dir_path)
            stop_imgs.extend(imgs)

    print(str(len(prohib_imgs) + len(peligro_imgs) + len(stop_imgs)) + ' imágenes cargadas.\n')
    return {"prohibicion": prohib_imgs, "peligro": peligro_imgs, "stop": stop_imgs}


def load_test_data(test_path):
    """Carga las imagenes de test con sus nombres.
        OUT: [img_nombre: img, ...] """

    print('Cargando imágenes de test...')

    imgs, img_names = __extract_imgs_in_dir(test_path)
    img_w_names = dict(zip(img_names, imgs))

    print(str(len(img_w_names)) + ' imágenes cargadas.\n')
    return img_w_names


def create_signs_mask(training_imgs):
    """Crea una máscara media para cada tipo de señal presente.
        OUT: [tipo_señal: mask, ...] """

    print('Creando máscaras media para cada tipo de señal...')
    final_imgs = {}

    for type_of_sign in training_imgs:
        # Cogemos todas las imagenes asociadas a un tipo de señal
        imgs = training_imgs[type_of_sign]

        # ----- Calculamos la máscara media ----- #
        for i in range(len(imgs)):
            # Por cada imagen calculamos su máscara
            img = cv2.resize(imgs[i], (25, 25))
            if i == 0:
                # Si es la primera iteración simplemente asignamos
                final_img = img
            else:
                # En caso de ser otra, calculamos la media entre la media de las mascaras anteriores y la actual
                alpha = 1.0 / (i + 1)
                beta = 1.0 - alpha
                final_img = cv2.addWeighted(img, alpha, final_img, beta, 0.0)

        # Asignamos al tipo de señal la máscara calculada
        final_imgs[type_of_sign] = __create_mask(final_img)

    print('Máscaras de señales creadas.\n')
    return final_imgs


def detect_regions(detector, test_imgs):
    """Asocia el nombre de una imagen con multiples regiones detectadas de esa imagen.
        OUT: [img_nombre: [Region1, Region2, ...], ...] """

    print('Detectando regiones en imágenes de test...')
    detected_regions = {}

    cont = 0
    if detector == 'mser' or detector == '':
        for img_name in test_imgs:
            mser_regions = __mser(test_imgs[img_name])
            cont += len(mser_regions)
            detected_regions[img_name] = mser_regions

    print(str(cont) + ' regiones detectadas y guardadas.\n')
    return detected_regions

    # elif otros posibles detectores ...


def delete_duplicates(detected_regions):
    """Elimina los duplicados comparando las imagenes según
    el valor del posición de sus píxeles."""

    print('Eliminando regiones duplicadas...')

    for img_nombre in detected_regions:
        regions = detected_regions[img_nombre]
        regions_no_dup = regions.copy()
        regions_len = len(regions)

        cont2 = 0
        for cont in range(regions_len):
            if cont + 1 < regions_len:
                mask_act = __create_mask(regions[cont].img)
                mask_next = __create_mask(regions[cont + 1].img)

                _, pos_score = __calculate_scores(mask_act, mask_next)

                if pos_score >= 500:
                    cont2 += 1
                    regions_no_dup.remove(regions[cont])

        detected_regions[img_nombre] = regions_no_dup

    print(str(cont2) + ' duplicados eliminados.\n')
    return detected_regions


def evaluate_regions(detected_regions, signs_masks, pixel_pos_mode=True):
    """Evalúa las regiones identificando las señales de tráfico."""

    print('Evaluando regiones...')

    max_score_prohib = signs_masks["prohibicion"].clip(max=1).sum()
    max_score_peligro = signs_masks["peligro"].clip(max=1).sum()
    max_score_stop = signs_masks["stop"].clip(max=1).sum()
    scores = [max_score_prohib, max_score_peligro, max_score_stop]

    cont = 0
    for img_nombre in detected_regions:
        regions = detected_regions[img_nombre]
        valid_regions = []
        cont += len(regions)

        for region in regions:
            region_mask = __create_mask(region.img)

            prohib_corr_score, prohib_pos_score = __calculate_scores(signs_masks["prohibicion"], region_mask)
            peligro_corr_score, peligro_pos_score = __calculate_scores(signs_masks["peligro"], region_mask)
            stop_corr_score, stop_pos_score = __calculate_scores(signs_masks["stop"], region_mask)
            sums = [prohib_corr_score, peligro_corr_score, stop_corr_score]

            prohib_dist = abs(max_score_prohib - sums[0])
            peligro_dist = abs(max_score_peligro - sums[1])
            stop_dist = abs(max_score_stop - sums[2])
            sign_distanc = [prohib_dist, peligro_dist, stop_dist]
            min_ind = sign_distanc.index(min(sign_distanc))

            # Si este modo esta activo entonces se toma en cuenta que los pixeles esten colocados de forma similar
            # que la máscara original
            sum2 = [prohib_pos_score, peligro_pos_score, stop_pos_score]
            pixel_pos = sum2[min_ind] > 400 if pixel_pos_mode else True

            if sign_distanc[min_ind] < 70 and pixel_pos:
                region.type = min_ind + 1
                region.score = sums[min_ind] * 100 / scores[min_ind]
                valid_regions.append(region)

        detected_regions[img_nombre] = valid_regions

    print(str(cont) + ' regiones evaluadas.\n')


def export_results(detected_regions):
    """Guarda los resultados en una carpeta y en ficheros de texto."""

    print('Guardando resultados...')
    new_dir = os.path.join(os.path.dirname(__file__), 'Resultados')

    try:
        os.remove("resultado.txt")
        os.remove("resultado_por_tipo.txt")
        shutil.rmtree(new_dir)
    except OSError:
        pass

    f1 = open("resultado.txt", "x")
    f2 = open("resultado_por_tipo.txt", "x")
    os.mkdir(new_dir)

    cont = 0
    for img_nombre in detected_regions:
        regions = detected_regions[img_nombre]
        for region in regions:
            cont += 1
            __write_txts(f1, f2, img_nombre, region)
            cv2.imwrite(os.path.join(new_dir, 'R' + str(cont) + '_' + img_nombre), region.img)

    f1.close()
    f2.close()
    print('Resultados guardados.')


# ------------------------------------------------------#
# ----------------  Auxiliar Methods   ---------------- #

def __extract_imgs_in_dir(dir_path):
    """Extrae imagenes ppm y jpg de un directorio"""

    imgs = []
    img_names = []
    for file in os.listdir(dir_path):
        if file.endswith(".ppm") or file.endswith(".jpg"):
            img = cv2.imread(dir_path + '/' + file)
            imgs.append(img)
            img_names.append(file)

    return imgs, img_names


def __create_mask(img):
    """Dada una imagen BGR crea una máscara 25x25"""

    img = cv2.resize(img, (25, 25))
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv_img, numpy.array([0, 50, 50]), numpy.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_img, numpy.array([170, 50, 50]), numpy.array([180, 255, 255]))
    final_mask = mask1 + mask2

    return final_mask


def __calculate_scores(mask_act, mask_next):
    """Calcula la correlacion y la posición de los píxeles que coinciden en cada imagen"""
    mask_act = mask_act.clip(max=1)
    mask_next = mask_next.clip(max=1)

    sum1 = sum2 = 0
    for i in range(25):
        for j in range(25):
            sum1 = sum1 + mask_act[i, j] * mask_next[i, j]
            if mask_act[i, j] == mask_next[i, j]:
                sum2 = sum2 + 1

    return sum1, sum2


def __mser(img):
    """Detecta distinas regiones de contraste mediante un algoritmo MSER"""

    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_delta=40, _max_variation=0.8)
    polygons = mser.detectRegions(bw_img)

    mser_regions = []

    if len(polygons[0]) == 0:
        bw_img = cv2.equalizeHist(bw_img)
        polygons = mser.detectRegions(bw_img)

    for polygon in polygons[0]:
        x, y, w, h = cv2.boundingRect(polygon)
        ratio = h / w if h / w >= 1 else w / h

        if ratio < 1.3:
            w2 = round(w * 1.5)  # agrandar dimensiones
            h2 = round(h * 1.5)
            x = round(x - (w2 - w) / 2)
            y = round(y - (h2 - h) / 2)

            try:
                crop = cv2.resize(img[y:y + h2, x:x + w2], (25, 25))
                region = Region(img=crop, coord=[y, y + h2, x, x + w2])
                mser_regions.append(region)
            except:
                pass

    return mser_regions


def __write_txts(f1, f2, img_nombre, region):
    f1.write(img_nombre + ";"
             + str(region.coord[0]) + ";" + str(region.coord[1]) + ";"
             + str(region.coord[2]) + ";" + str(region.coord[3]) + ";"
             + str(1) + ";" + str(region.score) + "\n")
    f2.write(img_nombre + ";"
             + str(region.coord[0]) + ";" + str(region.coord[1]) + ";"
             + str(region.coord[2]) + ";" + str(region.coord[3]) + ";"
             + str(region.type) + ";" + str(region.score) + "\n")
