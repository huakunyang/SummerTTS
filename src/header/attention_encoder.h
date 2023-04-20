#ifndef _ATTENTION_ENCODER_H_
#define _ATTENTION_ENCODER_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class attention_encoder
{
public:
    attention_encoder(float * modelData, int32_t & offset);
    ~attention_encoder();
    MatrixXf forward(const MatrixXf & x);
private:
    void * priv_;

};

#endif
