#ifndef _NN_CLAMP_MIN_H_
#define _NN_CLAMP_MIN_H_

#include <Eigen/Dense>
using Eigen::MatrixXf;

MatrixXf nn_clamp_min(const MatrixXf & x,float min);

#endif

