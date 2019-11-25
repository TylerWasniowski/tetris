FROM ubuntu:bionic

RUN apt update && \
apt install -y g++ make python3-dev python3-venv wget && \
mkdir /tetris-project && \
cd /tetris-project && \
wget -q https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.bz2 && \
tar --bzip2 -xf boost_1_71_0.tar.bz2 && \
cd boost_1_71_0 && \
./bootstrap.sh --with-libraries=python --with-python-version=3.6 --prefix=/usr/local && \
./b2 && \
echo "using python : 3.6 : /usr/bin/python3 : /usr/include/python3.6m : /usr/lib ;" >> ~/user-config.jam

RUN cd /tetris-project/boost_1_71_0 && \
./b2 install

RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

VOLUME /tetris-project/src
WORKDIR /tetris-project/src

# docker build -t tetris-project:latest .
# docker run --name tetris-project -t -i -v $PWD:/tetris-project/src tetris-project:latest
# docker start tetris-project
# docker exec -it tetris-project bash
# docker stop tetris-project
