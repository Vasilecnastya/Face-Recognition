import cv2
import numpy as np

import sys
import os
import signal

from variables import IMAGES_PATH, WEIGHTS_PATH, ALLOWED_EXTENSIONS


def process_arguments(args: list) -> str:
    if len(args) != 0 and args[0].split('.')[-1] in ALLOWED_EXTENSIONS:
        return 'file'
    return 'noargs'


def init_model() -> cv2.dnn:
    prototxt_path = os.path.join(WEIGHTS_PATH, 'deploy.prototxt.txt')
    model_path = os.path.join(WEIGHTS_PATH, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    return model


def show_result(image: np.ndarray, model_output: np.ndarray, height: int, width: int) -> None:
    font_scale = 1.0
    for i in range(0, model_output.shape[0]):
        confidence = model_output[i, 2]
        if confidence > 0.5:
            box = model_output[i, 3:7] * np.array([width, height, width, height])
            start_x, start_y, end_x, end_y = box.astype(np.int32)
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(0, 0, 255), thickness=2)
            cv2.putText(image, f"{confidence * 100:.2f}%", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,(0, 0, 255), 2)

    print('Для остановки выполнения нажмите на крестик')

    cv2.imshow("image", image)
    while True:
        cv2.waitKey(100)
        if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
            break


def process_file(model: cv2.dnn, filename: str) -> None:
    image = cv2.imread(filename)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    output = np.squeeze(model.forward())

    show_result(image, output, height, width)


if __name__ == '__main__':
    # Комментарий для провреки Github Actions (для коммита)
    args = sys.argv[1:]
    model = init_model()
    match (process_arguments(args)):
        case 'file':
            process_file(model, args[0])
        case 'noargs':
            process_file(model, os.path.join(IMAGES_PATH, 'test.png'))
    print('Обработка завершена')
