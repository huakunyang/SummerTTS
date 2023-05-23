#ifndef _FIX_DUR_PRED_H_
#define _FIX_DUR_PRED_H_

#include "DurationPredictor_base.h"
#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class FixDurationPredictor : public DurationPredictor_base
{
public:
    FixDurationPredictor(float * modelData, int32_t & offset,int32_t isMS);
    ~FixDurationPredictor();
    MatrixXf forward(const MatrixXf & x,const MatrixXf & g, float noiseScale);
    void setMSSpk(int32_t isMS, int32_t ginChannels);

private:
    void * priv_;

};

#endif
