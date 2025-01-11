cmake_minimum_required(VERSION 3.10)

# Set the project name
project(PigeonLearning)

# Add the executables
add_executable(pigeon_q_learning pigeon_q_learning.cpp)
add_executable(pigeon_deep_q_learning pigeon_deep_q_learning.cpp)
