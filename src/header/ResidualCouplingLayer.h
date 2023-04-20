#ifndef _RESIDUAL_COUPLING_LAYER_H_
#define _RESIDUAL_COUPLING_LAYER_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class ResidualCouplingLayer
{
public:
    ResidualCouplingLayer(float * modelData, int32_t & offset,int32_t dilation_rate,int32_t isMS);
    ~ResidualCouplingLayer();
    MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};

#endif
