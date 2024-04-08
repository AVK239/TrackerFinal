from datetime import datetime, timedelta

from flask import Flask, render_template, Response
import cv2
import torch
import time
import json
import os
from flask import jsonify
from win32timezone import now

app = Flask(__name__)

# Загрузка модели YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.15  # порог доверия
model.iou = 0.45  # порог IOU

video_path = 0  # Используйте 0 для веб-камеры
cap = cv2.VideoCapture(video_path)

timestamps = []
counts = []


def get_last_24_hours_data():
    filename = 'visitors_log.json'

    # Проверяем существование файла и создаем его, если не существует
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write("")  # Создаем пустой файл

    one_day_ago = datetime.now() - timedelta(days=1)
    filtered_data = []

    with open(filename, 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                record_time = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S')
                if record_time > one_day_ago:
                    filtered_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Ошибка при декодировании JSON: {e}")
            except ValueError as e:
                print(f"Ошибка при преобразовании даты: {e}")

    return filtered_data

def generate_frames():
    global timestamps, counts
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Обработка кадра моделью YOLO
            results = model(frame)
            last_logged_time = None
            # Подсчёт обнаруженных людей
            count_people = sum(1 for *_, cls in results.xyxy[0] if cls == 0)  # Класс 'человек'
            timestamps.append(time.time())
            counts.append(count_people)

            # Логируем результаты с определённой периодичностью
            if last_logged_time is None or (now - last_logged_time).total_seconds() > 60:  # Например, раз в минуту
                log_visitors(count_people)
                last_logged_time = now

            # Отрисовка рамок вокруг обнаруженных объектов
            for *xyxy, conf, cls in results.xyxy[0]:
                if cls == 0:  # класс 'человек'
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Функция для добавления записи о количестве посетителей
def log_visitors(count):
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'count': count
    }
    with open('visitors_log.json', 'a') as file:
        json.dump(record, file)
        file.write('\n')  # Добавляем новую строку для следующей записи

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    # Предполагается, что timestamps и counts обновляются в функции generate_frames
    global timestamps, counts
    # Очистка данных для избежания неограниченного роста списков
    if len(timestamps) > 1000:  # Число 1000 можно заменить на желаемое максимальное количество записей
        timestamps = timestamps[-1000:]
        counts = counts[-1000:]
    return jsonify({"timestamps": timestamps, "counts": counts})

@app.route('/data/last_24_hours')
def data_last_24_hours():
    data = get_last_24_hours_data()
    return jsonify(data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
