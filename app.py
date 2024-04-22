from datetime import datetime, timedelta
from threading import Thread

from flask import Flask, render_template, Response, request
import cv2
import torch
import time
import json
import schedule
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
visitors_buffer = []
timestamps = []
counts = []
correction_factor = 1.1


def get_aggregated_data():
    filename = 'visitors_log.json'
    one_day_ago = datetime.now() - timedelta(days=1)
    hourly_data = {i: [] for i in range(24)}  # Подготовка словаря для данных по часам

    # Проверка на существование файла
    if not os.path.exists(filename):
        return []

    with open(filename, 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                timestamp = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S')
                if timestamp > one_day_ago:
                    hour = timestamp.hour
                    hourly_data[hour].append(record['count'])
            except json.JSONDecodeError as e:
                print(f"Ошибка при декодировании JSON: {e}")
            except ValueError as e:
                print(f"Ошибка при преобразовании даты: {e}")

    # Агрегация данных
    aggregated_data = []
    for hour, counts in hourly_data.items():
        if counts:
            avg_count = sum(counts) / len(counts)
            aggregated_data.append({'hour': hour, 'average_count': avg_count})
        else:
            aggregated_data.append({'hour': hour, 'average_count': 0})

    return aggregated_data


def generate_frames():
    global timestamps, counts
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Обработка кадра моделью YOLO
            results = model(frame)

            # Подсчёт обнаруженных людей и применение коррекции
            raw_count_people = sum(1 for *_, cls in results.xyxy[0] if cls == 0)  # Класс 'человек'
            corrected_count_people = int(round(raw_count_people * correction_factor))  # Применяем коррекцию

            # Обновляем глобальные переменные
            timestamps.append(time.time())
            counts.append(corrected_count_people)

            last_logged_time = None
            if last_logged_time is None or (datetime.now() - last_logged_time).total_seconds() > 60:
                log_visitors(corrected_count_people)  # Логируем скорректированное количество
                last_logged_time = datetime.now()

            # Отрисовка рамок вокруг обнаруженных объектов
            for *xyxy, conf, cls in results.xyxy[0]:
                if cls == 0:  # класс 'человек'
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Функция для добавления записи о количестве посетителей
def log_visitors(count):
    """
    Добавляет информацию о количестве посетителей в буфер.
    """
    global visitors_buffer
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'count': count
    }
    visitors_buffer.append(record)


def aggregate_and_save_data():
    global visitors_buffer
    if not visitors_buffer:
        return

    # Группируем данные по часам
    hourly_aggregated_data = {}
    for record in visitors_buffer:
        hour = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S').hour
        if hour not in hourly_aggregated_data:
            hourly_aggregated_data[hour] = []
        hourly_aggregated_data[hour].append(record['count'])

    # Агрегируем данные для каждого часа
    aggregated_records = []
    for hour, counts in hourly_aggregated_data.items():
        average_count = sum(counts) / len(counts)
        timestamp = datetime.now().replace(minute=0, second=0, microsecond=0, hour=hour).strftime('%Y-%m-%d %H:%M:%S')
        aggregated_records.append({'timestamp': timestamp, 'average_count': average_count})

    # Сохраняем агрегированные данные в файл
    with open('aggregated_visitors_log.json', 'a') as file:
        for record in aggregated_records:
            json.dump(record, file)
            file.write('\n')

    # Очищаем буфер
    visitors_buffer.clear()


def periodic_aggregation_task():
    """
    Выполняет агрегацию и сохранение данных каждую минуту.
    """
    while True:
        aggregate_and_save_data()
        time.sleep(60)  # Пауза на 60 секунд

# Запускаем фоновый процесс агрегации данных
aggregation_thread = Thread(target=periodic_aggregation_task)
aggregation_thread.daemon = True
aggregation_thread.start()

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
    aggregated_data = get_aggregated_data()  # Используйте новую функцию агрегации
    return jsonify(aggregated_data)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data/archive')
def get_archive_data():
    selected_date = request.args.get('date')  # Получаем дату из параметров запроса
    if not selected_date:
        return jsonify({"error": "Не указана дата"}), 400

    selected_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
    data = []

    with open('visitors_log.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                record_date = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S').date()
                if record_date == selected_date:
                    data.append(record)
            except json.JSONDecodeError as e:
                print(f"Ошибка при декодировании JSON: {e}")
            except ValueError as e:
                print(f"Ошибка при преобразовании даты: {e}")



    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)