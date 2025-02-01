cmake_minimum_required(VERSION 3.10)

# Set the project name
project(QLearning)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Add the executables
add_executable(pigeon_q_learning pigeon_q_learning.cpp)
add_executable(pigeon_deep_q_learning pigeon_deep_q_learning.cpp)
add_executable(pigeon_double_deep_q_learning pigeon_double_deep_q_learning.cpp)
add_executable(taxi_q_learning taxi_q_learning.cpp)
add_executable(taxi_q_learning_random taxi_q_learning_random.cpp)
add_executable(multi_layer multi_layer.cpp)
add_executable(transformer transformer.cpp)
