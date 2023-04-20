#ifndef _H_CONV_FLOW_H_
#define _H_CONV_FLOW_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class ConvFlow
{
public:
    ConvFlow(float * modelData, int32_t & offset);
    ~ConvFlow();
    MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};

#endif
