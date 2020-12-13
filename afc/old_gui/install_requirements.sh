#!/bin/bash
# Install User Interface requirements (easier)
apt-get  update \
        && apt-get upgrade\
		&& apt-get -y install python3 python3-pip
pip3 install -r requirements.txt
echo
echo
echo OK: openCV 4.1.0, PyQ5t, numpy, requests and imutils installed successfully.
echo
echo To start the user Interface, execute:
echo 'python3 user_interface.py  -dir <images_dir>  -user <username>  -pass <password>  -url <url>  -mission <mission_number>'
echo
echo