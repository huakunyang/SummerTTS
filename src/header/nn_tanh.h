#ifndef _NN_TANH_H_
#define _NN_TANH_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

MatrixXf nn_tanh(const MatrixXf & x);

#endif
