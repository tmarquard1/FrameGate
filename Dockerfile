FROM debian:10

WORKDIR /home
ENV HOME /home
RUN cd ~
RUN apt-get update
RUN apt-get install -y git nano python3-pip python-dev pkg-config wget usbutils curl 

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
| tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y edgetpu-examples 
RUN apt-get install -y libedgetpu1-std
RUN apt-get install -y python3-pycoral
ENTRYPOINT ["tail", "-f", "/dev/null"]

#python3 /usr/share/edgetpu/examples/classify_image.py --model /usr/share/edgetpu/examples/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --label /usr/share/edgetpu/examples/models/inat_bird_labels.txt --image /usr/share/edgetpu/examples/images/bird.bmp

# sudo echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTR{idProduct}=="9302", MODE="0666"' > /etc/udev/rules.d/90-usb-rtlsdr.rules