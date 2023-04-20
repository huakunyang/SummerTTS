#ifndef _NN_CUMSUM_H_
#define _NN_CUMSUM_H_

#include <Eigen/Dense>
using Eigen::MatrixXf;

MatrixXf nn_cumsum(const MatrixXf & x, int32_t dim);

#endif
