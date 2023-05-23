#ifndef _DUR_PRED_H_
#define _DUR_PRED_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class DurationPredictor_base
{
public:
    virtual ~DurationPredictor_base() = 0;
    virtual MatrixXf forward(const MatrixXf & x,const MatrixXf & g, float noiseScale) = 0;
    virtual void setMSSpk(int32_t isMS, int32_t ginChannels) = 0;

private:
    void * priv_;

};

#endif
