#ifndef _RESBLOCK1_H_
#define _RESBLOCK1_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class ResBlock1
{
public:
    ResBlock1(float * modelData, int32_t & offset);
    ~ResBlock1();
    MatrixXf forward(const MatrixXf & x);

private:
    void * priv_;

};

#endif
