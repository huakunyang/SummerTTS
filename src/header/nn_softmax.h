#ifndef _NN_OP_SOFTMAX_H_
#define _NN_OP_SOFTMAX_H_

#include <Eigen/Dense>
using Eigen::MatrixXf;

MatrixXf nn_softmax(const MatrixXf & mat, int32_t dim);

#endif
