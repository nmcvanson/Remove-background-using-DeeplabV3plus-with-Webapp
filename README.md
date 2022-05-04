# Удаление фона с помощью нейтронных сетей

В данной работе разработан подход к удалению фона человеческих изображений. Основная идея состоит в том, что мы бужем сегментировать человеческие изображения с помощью архитектуры DeepLabV3+. Весь код написан на Питоне, с использованием TensorFlow 2.5.0.

## Содержание

- Семантическая сегментация
- Person Segmentation Dataset
- Реализован подход к обработке данных
- Построена структура Deeplabv3+
- Обучена модель
- Оценка по модели
- Реализован простейший веб интерфейс для удаления фона

## Семантическая сегментация

Семантическая сегментация - хорошо известная задача компьютерного зрения, одна из трех важнейших, наряду с классификацией и обнаружением объектов. Сегментация, на самом деле, является задачей классификации, в смысле распределения каждого пикселя по классам. В отличие от моделей классификации или обнаружения изображений, модель сегментации действительно демонстрирует некоторое «понимание» изображений, то есть не только говорит, что «на этом изображении есть кошка», но и на уровне пикселей указывает, где эта кошка.  
Одна из самых простых и популярных архитектур, используемых для семантической сегментации, это полносверточная сеть (Fully Convolutional Network, FCN). В статье "Fully Convolutional Networks for Semantic Segmentation" (<https://arxiv.org/pdf/1411.4038.pdf>) авторы используют FCN для первоначального преобразования входного изображения до меньшего размера (одновременно с этим увеличивая количество каналов) через серию сверток. Такой набор сверточных операций обычно называется кодировщик. Затем выход декодируется или через билинейную интерполяцию, или через серию транспонированных сверток, который носит название декодер.
<img src="https://cdn-images-1.medium.com/max/1200/1*edkNzGBBDBXtpZMq-pnSng.png =240x120" width=50% height=50%>



