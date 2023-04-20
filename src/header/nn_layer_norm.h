#ifndef _NN_LAYER_NORM_H_
#define _NN_LAYER_NORM_H_

#include <Eigen/Dense>
#include "stdint.h"

using Eigen::MatrixXf;

class nn_layer_norm
{
public:
    nn_layer_norm(float * modelData, int32_t & offset);
    nn_layer_norm(int32_t size, const MatrixXf & gamma, const MatrixXf & beta);
    MatrixXf forward(const MatrixXf & x);

private:
    void * priv_;

};

#endif
