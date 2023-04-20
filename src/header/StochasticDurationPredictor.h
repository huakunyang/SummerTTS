#ifndef _STOCH_DUR_PRED_H_
#define _STOCH_DUR_PRED_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class StochasticDurationPredictor
{
public:
    StochasticDurationPredictor(float * modelData, int32_t & offset,int32_t isMS);
    ~StochasticDurationPredictor();
    MatrixXf forward(const MatrixXf & x,const MatrixXf & g, float noiseScale);
    void setMSSpk(int32_t isMS, int32_t ginChannels);

private:
    void * priv_;

};

#endif
