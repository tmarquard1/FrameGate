FROM python:3.10

WORKDIR /home
ENV HOME /home


RUN wget -O kinetics_600_labels.txt https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt

# Download and save the movinet model using Python
RUN python3 -c "\
import tensorflow as tf; \
import tensorflow_hub as hub; \
from pathlib import Path; \
model_id = 'a2'; \
model_mode = 'stream'; \
model_version = '3'; \
hub_url = f'https://tfhub.dev/tensorflow/movinet/{model_id}/{model_mode}/kinetics-600/classification/{model_version}'; \
local_model_path = Path('/home/movinet_model'); \
local_model_path.mkdir(parents=True, exist_ok=True); \
model = hub.load(hub_url); \
tf.saved_model.save(model, str(local_model_path))"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Verify installation
RUN python3 --version && pip --version

COPY coral/inference.py coral/inference.py

RUN mkdir -p /Downloads

EXPOSE 8000
# uvicorn coral.inference:app --host 0.0.0.0 --port 8000 --reload
ENTRYPOINT [ "uvicorn", "coral.inference:app", "--host", "0.0.0.0", "--port", "8000", "--reload" ]

#ENTRYPOINT ["tail", "-f", "/dev/null"]

# Example command to run the model
# python3 /usr/share/edgetpu/examples/classify_image.py --model /usr/share/edgetpu/examples/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --label /usr/share/edgetpu/examples/models/inat_bird_labels.txt --image /usr/share/edgetpu/examples/images/bird.bmp

# Example udev rule for USB devices
# sudo echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTR{idProduct}=="9302", MODE="0666"' > /etc/udev/rules.d/90-usb-rtlsdr.rules

# Example environment variable for picamera installation
# export READTHEDOCS=True for getting picamera to pip install
# pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime