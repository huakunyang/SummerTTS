#ifndef _H_DDS_CONV_H_
#define _H_DDS_CONV_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class DDSConv
{
public:
    DDSConv(float * modelData, int32_t & offset);
    ~DDSConv();
    MatrixXf forward(const MatrixXf & x, const MatrixXf & g, int32_t hasG);

private:
    void * priv_;

};

#endif
