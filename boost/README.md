* To build Docker image from Dockerfile:
  * docker build -t tetris-project:latest .
* To run Docker container from image:
  * docker run --name tetris-project -t -i -v $PWD:/tetris-project/src tetris-project:latest
* Once inside Docker container, to compile C++ program:
  * cd /tetris-project/src/tetris/boost
  * ./compile.sh
* To run Python program:
  * cd /tetris-project/src/tetris/boost
  * ./run.sh
* Or to compile and run with a single script:
  * cd /tetris-project/src/tetris/boost
  * ./compile-run.sh
