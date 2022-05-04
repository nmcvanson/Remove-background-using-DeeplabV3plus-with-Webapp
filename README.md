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

<p align="justify">Семантическая сегментация - хорошо известная задача компьютерного зрения, одна из трех важнейших, наряду с классификацией и обнаружением объектов. Сегментация, на самом деле, является задачей классификации, в смысле распределения каждого пикселя по классам. В отличие от моделей классификации или обнаружения изображений, модель сегментации действительно демонстрирует некоторое «понимание» изображений, то есть не только говорит, что «на этом изображении есть кошка», но и на уровне пикселей указывает, где эта кошка.</p>
<p align="justify">Одна из самых простых и популярных архитектур, используемых для семантической сегментации, это полносверточная сеть (Fully Convolutional Network, FCN). В статье <a href = "https://arxiv.org/pdf/1411.4038.pdf"> "Fully Convolutional Networks for Semantic Segmentation" </a> авторы используют FCN для первоначального преобразования входного изображения до меньшего размера (одновременно с этим увеличивая количество каналов) через серию сверток. Такой набор сверточных операций обычно называется кодировщик. Затем выход декодируется или через билинейную интерполяцию, или через серию транспонированных сверток, который носит название декодер.</p>
<p align="center">
<img src="https://cdn-images-1.medium.com/max/1200/1*edkNzGBBDBXtpZMq-pnSng.png" width="480" height="240">
</p>
<p align="center">Структура FCN</p>
<p align="justify"><a href = "https://arxiv.org/pdf/1505.04597.pdf">Сеть U-Net</a> представляет из себя улучшение простой FCN архитектуры. Сеть skip-связи между выходами с блоков свертки и соответствующими им входами блока транспонированной свертки на том же уровне. Skip-связи позволяют градиентам лучше распространяться и предоставлять информацию с различных масштабов размера изображения. Информация с больших масштабов (верхние слои) может помочь модели лучше классифицировать. В то время как информация с меньших масштабов (глубокие слои) помогает модели лучше сегментировать.</p>
<p align="center">
<img src="https://cdn-images-1.medium.com/max/1200/1*DxXHzO7JIZs24g-UQmZ8Hw.png" width="480" height="240">
</p>
<p align="center">Структура U-NET</p>
