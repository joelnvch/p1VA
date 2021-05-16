import cv2
import numpy
import os
from matplotlib import pyplot as plt


class Region:
    def __init__(self, img, coord):
        self.img = img
        self.coord = coord
        self.type = None
        self.score = 0


def load_training_data(train_path):
    """Carga las imágenes de entrenamiento devolviendolas agrupadas según su tipo.
        OUT: [tipo_señal: [img1, img2, ...], ...] """

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

    return {"prohibicion": prohib_imgs, "peligro": peligro_imgs, "stop": stop_imgs}


def load_test_data(test_path):
    """Carga las imagenes de test con sus nombres.
        OUT: [img_nombre: img, ...] """

    imgs, img_names = __extract_imgs_in_dir(test_path)
    img_w_names = dict(zip(img_names, imgs))

    return img_w_names


def create_signs_mask(training_imgs):
    """Crea una máscara media para cada tipo de señal presente.
        OUT: [tipo_señal: mask, ...] """

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

    return final_imgs


def detect_regions(detector, test_imgs):
    """Asocia el nombre de una imagen con multiples regiones detectadas de esa imagen.
        OUT: [img_nombre: [Region1, Region2, ...], ...] """

    detected_regions = {}

    if detector == 'mser':
        for img_name in test_imgs:
            mser_regions = __mser(test_imgs[img_name])
            detected_regions[img_name] = mser_regions

        return detected_regions

    # elif otros posibles detectores ...


def delete_duplicates(detected_regions):
    """Elimina los duplicados comparando las imagenes según
    el valor del posición de sus píxeles."""

    for img_nombre in detected_regions:
        regions = detected_regions[img_nombre]
        regions_no_dup = regions.copy()
        regions_len = len(regions)

        for cont in range(regions_len):
            if cont + 1 < regions_len:
                mask_act = __create_mask(regions[cont].img)
                mask_next = __create_mask(regions[cont + 1].img)

                sum = 0
                for i in range(25):
                    for j in range(25):
                        if mask_act[i, j] == mask_next[i, j]:
                            sum = sum + 1

                if sum >= 500:
                    regions_no_dup.remove(regions[cont])

        detected_regions[img_nombre] = regions_no_dup

    return detected_regions


def evaluate_regions(detected_regions, signs_masks):
    prohib_mask = signs_masks["prohibicion"].clip(max=1)
    peligro_mask = signs_masks["peligro"].clip(max=1)
    stop_mask = signs_masks["stop"].clip(max=1)

    for img_nombre in detected_regions:
        regions = detected_regions[img_nombre]
        valid_regions = []
        for region in regions:
            mask = __create_mask(region.img).clip(max=1)
            prohib_sum = peligro_sum = stop_sum = 0
            prohib_sum2 = peligro_sum2 = stop_sum2 = 0

            for i in range(25):
                for j in range(25):
                    prohib_sum = prohib_sum + mask[i, j] * prohib_mask[i, j]
                    peligro_sum = peligro_sum + mask[i, j] * peligro_mask[i, j]
                    stop_sum = stop_sum + mask[i, j] * stop_mask[i, j]

                    if mask[i, j] == prohib_mask[i, j]:
                        prohib_sum2 = prohib_sum2 + 1
                    if mask[i, j] == peligro_mask[i, j]:
                        peligro_sum2 = peligro_sum2 + 1
                    if mask[i, j] == stop_mask[i, j]:
                        stop_sum2 = stop_sum2 + 1

            sums = [prohib_sum, peligro_sum, stop_sum]
            sum2 = [prohib_sum2, peligro_sum2, stop_sum2]
            max_val_ind = sums.index(max(sums))
            max_val_ind2 = sum2.index(max(sum2))
            
            if (sums[0] >= 90 or sums[1] >= 90 or sums[2] >= 200) and sum2[max_val_ind2] >= 425:
                print(img_nombre)
                print(sums)
                # mas cerca de 200 o 120
                a = abs(200 - sums[2])
                b = (min(abs(130 - sums[1]), abs(130 - sums[0])))
                if a < b:
                    region.type = 3
                elif abs(130 - sums[0]) > abs(130 - sums[1]):
                    region.type = 2
                else:
                    region.type = 1

                region.score = sums[max_val_ind]
                valid_regions.append(region)

        detected_regions[img_nombre] = valid_regions


def export_results(detected_regions):
    try:
        f1 = open("resultado.txt", "x")
    except:
        os.remove("resultado.txt")
        f1 = open("resultado.txt", "x")

    try:
        f2 = open("resultado_por_tipo.txt", "x")
    except:
        os.remove("resultado_por_tipo.txt")
        f2 = open("resultado_por_tipo.txt", "x")

    for img_nombre in detected_regions:
        regions = detected_regions[img_nombre]

        for region in regions:
            f1.write(img_nombre + ";"
                    + str(region.coord[0]) + ";" + str(region.coord[1]) + ";"
                    + str(region.coord[2]) + ";" + str(region.coord[3]) + ";"
                    + str(1) + ";" + str(region.score) + "\n")

            f2.write(img_nombre + ";"
                    + str(region.coord[0]) + ";" + str(region.coord[1]) + ";"
                    + str(region.coord[2]) + ";" + str(region.coord[3]) + ";"
                    + str(region.type) + ";" + str(region.score) + "\n")

    f1.close()
    f2.close()


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


def __mser(img):
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bw_img = cv2.equalizeHist(bw_img)

    mser = cv2.MSER_create(_delta=40, _max_variation=0.8)
    polygons = mser.detectRegions(bw_img)

    mser_regions = []
    for polygon in polygons[0]:
        x, y, w, h = cv2.boundingRect(polygon)
        ratio = h / w if h / w >= 1 else w / h

        if ratio < 1.3:
            w2 = round(w * 1.4)  # agrandar dimensiones
            h2 = round(h * 1.4)
            x = round(x - (w2 - w) / 2)
            y = round(y - (h2 - h) / 2)

            try:
                crop = cv2.resize(img[y:y + h2, x:x + w2], (25, 25))
                region = Region(img=crop, coord=[y, y + h2, x, x + w2])
                mser_regions.append(region)
            except:
                pass

    return mser_regions
