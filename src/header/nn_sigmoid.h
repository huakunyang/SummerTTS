#ifndef _NN_SIGMOID_H_
#define _NN_SIGMOID_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

MatrixXf nn_sigmoid(const MatrixXf & x);

#endif
