#ifndef _WN_H_
#define _WN_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class WN
{
public:
    WN(float * modelData, int32_t & offset, int32_t dilation_rate,int32_t isMS);
    ~WN();
    MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};
#endif
