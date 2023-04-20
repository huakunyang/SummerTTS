#ifndef _TEXT_ENCODER_H_
#define _TEXT_ENCODER_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class TextEncoder
{
public:
    TextEncoder(float * modelData, int32_t & offset);
    MatrixXf forward(int32_t * x, int32_t length, MatrixXf & m, MatrixXf & logs);
    ~TextEncoder();

private:
    void * priv_;
};

#endif
