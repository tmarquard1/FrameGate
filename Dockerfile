FROM python:3.9

WORKDIR /home
ENV HOME /home

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    nano \
    pkg-config \
    wget \
    usbutils \
    curl \
    gnupg \
    software-properties-common \
    gdal-bin \
    libgdal-dev

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Add Coral Edge TPU repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update

# Install Edge TPU runtime
RUN apt-get install -y libedgetpu1-std

# RUN apt-get install -y python3-pycoral
# Install python3-pycoral and tflite_runtime using pip
RUN pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
RUN pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_aarch64.whl

# Download the mobilenet model
RUN wget -O mobilenet_v1_1.0_224_edgetpu.tflite https://dl.google.com/coral/canned_models/mobilenet_v1_1.0_224_quant_edgetpu.tflite
RUN wget -O imagenet_labels.txt https://dl.google.com/coral/canned_models/imagenet_labels.txt

RUN wget -O mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite
RUN wget -O inat_bird_labels.txt https://dl.google.com/coral/canned_models/inat_bird_labels.txt



COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Verify installation
RUN python3 --version && pip --version

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY coral/inference.py coral/inference.py

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