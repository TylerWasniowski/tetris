PYTHON_VERSION = 3.6
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
# PYTHON_VERSION = 3.6m
# PYTHON_INCLUDE =  /home/twasniowski/.conda/envs/tetris/include/python$(PYTHON_VERSION)

# BOOST_INC = /home/twasniowski/.conda/envs/tetris/include/boost
# BOOST_LIB = /home/twasniowski/.conda/envs/tetris/lib
BOOST_INC = /tetris-boost/boost_1_71_0
BOOST_LIB = $(BOOST_INC)/stage/lib

TARGET = tetris

CXX = c++17

# GCC = ~/gentoo/tmp/usr/bin/gcc
# GCC = g++-4.8
GCC = g++

boost:
	$(GCC) -std=$(CXX) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TARGET).cpp
	$(GCC) -std=$(CXX) -shared -Wall -Werror -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python36 -o $(TARGET).so

# $(TARGET).so: $(TARGET).o
# 	g++ -shared -Wall -Werror -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python36 -o $(TARGET).so

# $(TARGET).o: $(TARGET).cpp
# 	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TARGET).cpp

cpp:
	$(GCC) -o $(TARGET) $(TARGET).cpp

clean:
	rm -f $(TARGET).o $(TARGET).so
