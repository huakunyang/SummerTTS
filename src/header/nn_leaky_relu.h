#ifndef _NN_LEAKY_RELU_H_
#define _NN_LEAKY_RELU_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

MatrixXf nn_leaky_relu(const MatrixXf & x);
MatrixXf nn_leaky_relu(const MatrixXf & x, float slope);

#endif
