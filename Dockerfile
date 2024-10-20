FROM python:3.8

WORKDIR /home
ENV HOME /home
RUN cd ~
RUN apt-get update
RUN apt-get install -y git nano pkg-config wget usbutils curl gnupg

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
| tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
# RUN apt-get install -y edgetpu-examples 
RUN apt-get install -y libedgetpu1-std
# Install python3-tflite-runtime manually
#RUN apt-get install -y python3-tflite-runtime=2.5.0.post1
RUN apt-get install -y python3-pycoral
ENTRYPOINT ["tail", "-f", "/dev/null"]

#python3 /usr/share/edgetpu/examples/classify_image.py --model /usr/share/edgetpu/examples/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --label /usr/share/edgetpu/examples/models/inat_bird_labels.txt --image /usr/share/edgetpu/examples/images/bird.bmp

# sudo echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTR{idProduct}=="9302", MODE="0666"' > /etc/udev/rules.d/90-usb-rtlsdr.rules

# export READTHEDOCS=True for getting picamera to pip install
# pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime