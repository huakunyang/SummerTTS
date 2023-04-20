#ifndef _NN_GELU_H_
#define _NN_GELU_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

MatrixXf nn_gelu(const MatrixXf & x);

#endif
