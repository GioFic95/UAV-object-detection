#!/bin/bash
# Install User Interface requirements
apt-get  update \
        && apt-get upgrade\
		&& apt-get -y install\
		wget nano unzip \
		python\
		python3\
        python3-pyqt5 \
        build-essential cmake unzip pkg-config \
        libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev \
        libgtk-3-dev \
        libatlas-base-dev gfortran \
        python-dev python3-dev \
        python3-pip
#pip --yes uninstall numpy # tolgo numpy da python2 con pip perchè quando installa opencv così trova solo quello di python3
pip3 install 'imutils==0.5.2' # se non trova questa, trovarne una non troppo distante come data
pip3 install 'numpy==1.16.3' # se non trova questa, trovarne una non troppo distante come data
pip3 install requests
cd $HOME
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.1.0 opencv
mv opencv_contrib-4.1.0 opencv_contrib
rm opencv.zip
rm opencv_contrib.zip
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON ..
make -j6
make install
#pip install numpy
echo
echo
echo OK: openCV 4.1.0, PyQ5t, numpy, requests and imutils installed successfully.
echo
echo To start the user Interface, execute:
echo 'python3 user_interface.py  -dir <images_dir>  -user <username>  -pass <password>  -url <url>  -mission <mission_number>'
echo
echo