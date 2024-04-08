import torch
import torch.onnx

# Загрузите предварительно обученную модель YOLOv5
model = torch.load('yolov5s.pt')
model.eval()

# Создайте тензор для входных данных. Размеры могут отличаться в зависимости от модели.
# Например, для YOLOv5s размер входного изображения обычно составляет 640x640.
x = torch.randn(1, 3, 640, 640, requires_grad=True)

# Укажите путь для сохранения модели ONNX
output_onnx = 'yolov5.onnx'

# Экспорт модели
torch.onnx.export(model,               # модель
                  x,                   # модель входа (или кортеж для нескольких входов)
                  output_onnx,         # где сохранить модель (может быть файловый объект)
                  export_params=True,  # сохранить обученные параметры веса внутри модели
                  opset_version=10,    # версия ONNX, должна быть совместима с вашими инструментами
                  do_constant_folding=True,  # оптимизация: удаляет постоянные складки для улучшения оптимизации
                  input_names = ['input'],   # имя входов
                  output_names = ['output'], # имя выходов
                  dynamic_axes={'input' : {0 : 'batch_size'},    # переменный размер пакета
                                'output' : {0 : 'batch_size'}})
