#ifndef _RESIDUAL_COUPLING_BLOCK_H_
#define _RESIDUAL_COUPLING_BLOCK_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class ResidualCouplingBlock
{
public:
    ResidualCouplingBlock(float * modelData, int32_t & offset,int32_t dilation_rate,int32_t isMS);
    ~ResidualCouplingBlock();
    MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};

#endif
