FROM python:3.8
RUN pip install poetry

WORKDIR /code
COPY . .

RUN poetry config virtualenvs.create false
RUN poetry config installer.max-workers 1
RUN poetry install

RUN pip install -U 'tensorboardX'
RUN pip install -U 'tensorboard'
RUN pip install opencv-python
RUN pip install wandb
RUN pip install pycocotools
RUN pip install gdown
RUN apt-get update
RUN apt-get install ffmpeg libgl1 xvfb -y