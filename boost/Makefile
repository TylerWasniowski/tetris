PYTHON_VERSION = 3.6
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
# PYTHON_VERSION = 3.6m
# PYTHON_INCLUDE =  /home/twasniowski/.conda/envs/tetris/include/python$(PYTHON_VERSION)

BOOST_INC = /tetris-project/boost_1_71_0
BOOST_LIB = $(BOOST_INC)/stage/lib
# BOOST_INC = /home/twasniowski/.conda/envs/tetris/include/boost
# BOOST_LIB = /home/twasniowski/.conda/envs/tetris/lib

# TETRIS = tetris_boost
TETRIS = tetris_boost_5

CXX = c++17

GCC = g++
# GCC = ~/gentoo/tmp/usr/bin/gcc

boost:
	$(GCC) -std=$(CXX) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TETRIS).cpp
	$(GCC) -std=$(CXX) -shared -Wall -Werror -Wl,--export-dynamic $(TETRIS).o -L$(BOOST_LIB) -lboost_python36 -o $(TETRIS).so

# $(TETRIS).so: $(TETRIS).o
# 	g++ -shared -Wall -Werror -Wl,--export-dynamic $(TETRIS).o -L$(BOOST_LIB) -lboost_python36 -o $(TETRIS).so

# $(TETRIS).o: $(TETRIS).cpp
# 	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -w $(TETRIS).cpp

cpp:
	$(GCC) -o $(TETRIS) $(TETRIS).cpp

clean:
	rm -f $(TETRIS).o $(TETRIS).so
