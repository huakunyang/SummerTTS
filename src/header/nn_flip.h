#ifndef _NN_FLIP_H_
#define _NN_FLIP_H_

#include <Eigen/Dense>
using Eigen::MatrixXf;

MatrixXf nn_flip(const MatrixXf & x,int32_t dim);

#endif
