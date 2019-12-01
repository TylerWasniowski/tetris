PYTHON_VERSION = 3.6m
#PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
PYTHON_INCLUDE =  /home/twasniowski/.conda/envs/tetris/include/python$(PYTHON_VERSION)

BOOST_INC = /home/twasniowski/tetris/boost/boost_1_71_0
BOOST_LIB = $(BOOST_INC)/stage/lib

TARGET = tetris

CXX = c++11

boost:
	g++ -std=$(CXX) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TARGET).cpp
	g++ -std=$(CXX) -shared -Wall -Werror -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python36 -o $(TARGET).so

# $(TARGET).so: $(TARGET).o
# 	g++ -shared -Wall -Werror -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python36 -o $(TARGET).so

# $(TARGET).o: $(TARGET).cpp
# 	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TARGET).cpp

cpp:
	g++ -o $(TARGET) $(TARGET).cpp

clean:
	rm -f $(TARGET).o $(TARGET).so
