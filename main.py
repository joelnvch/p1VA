import argparse

from signs_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # ----------------  Datos Prueba   ---------------- #
    args.train_path = 'C:/Users/jn.villaverde.2017/Downloads/train_recortadas/train_recortadas'
    args.test_path = 'C:/Users/jn.villaverde.2017/Downloads/test/test'
    args.detector = 'mser'
    # ----------------  Datos Prueba   ---------------- #

    # Cargar imágenes
    training_imgs = load_training_data(args.train_path)
    test_imgs = load_test_data(args.test_path)

    # Procesar las imágenes
    signs_masks = create_signs_mask(training_imgs)
    detected_regions = detect_regions(args.detector, test_imgs)
    delete_duplicates(detected_regions)

    # Evaluar los resultados e imprimirlos en .txt
    evaluate_regions(detected_regions, signs_masks)
    export_results(detected_regions)
