PYTHON_VERSION = 3.6m
# PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
PYTHON_INCLUDE =  /home/twasniowski/.conda/envs/tetris/include/python$(PYTHON_VERSION)

BOOST_INC = /home/twasniowski/.conda/envs/tetris/include/boost
BOOST_LIB = /home/twasniowski/.conda/envs/tetris/lib

TARGET = tetris

CXX = c++17

boost:
	~/gentoo/tmp/usr/bin/gcc -std=$(CXX) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TARGET).cpp
	~/gentoo/tmp/usr/bin/gcc -std=$(CXX) -shared -Wall -Werror -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python36 -o $(TARGET).so

# $(TARGET).so: $(TARGET).o
# 	g++ -shared -Wall -Werror -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python36 -o $(TARGET).so

# $(TARGET).o: $(TARGET).cpp
# 	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TARGET).cpp

cpp:
	~/gentoo/tmp/usr/bin/gcc -o $(TARGET) $(TARGET).cpp

clean:
	rm -f $(TARGET).o $(TARGET).so
