PYTHON_VERSION = 3.6
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

BOOST_INC = /tetris-project/boost_1_71_0
BOOST_LIB = $(BOOST_INC)/stage/lib

TARGET = tetris

$(TARGET).so: $(TARGET).o
	g++ -shared -Wall -Werror -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python36 -o $(TARGET).so

$(TARGET).o: $(TARGET).cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TARGET).cpp

clean:
	rm -f $(TARGET).o $(TARGET).so
