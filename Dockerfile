FROM ubuntu:16.04

ADD l_openvino_toolkit* /openvino/

ARG INSTALL_DIR=/opt/intel/openvino 

RUN apt-get update && apt-get -y upgrade && apt-get autoremove

#Install needed dependences
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cpio \
        curl \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3.5-dev \
        python3-pip \
        python3-setuptools \
        python-pil \
        sudo \
        udev \
        usbutils

# installing OpenVINO dependencies
RUN cd /openvino/ && \
    ./install_openvino_dependencies.sh

RUN pip3 install numpy

# installing OpenVINO itself
RUN cd /openvino/ && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh --silent silent.cfg

# Model Optimizer
RUN cd $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites && \
    ./install_prerequisites.sh

RUN cd ${INSTALL_DIR}/install_dependencies && \
    ./install_NCS_udev_rules.sh

# clean up 
RUN apt autoremove -y && \
    rm -rf /openvino /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && pip install matplotlib flask pillow

# CMD ["/bin/bash"]
EXPOSE 5000
WORKDIR /opt/app
RUN echo "source $INSTALL_DIR/bin/setupvars.sh" >> /root/.bashrc
ENTRYPOINT ["/opt/app/entrypoint.sh"]