#ifndef _GENERATOR_HIFI_GAN_H_
#define _GENERATOR_HIFI_GAN_H_

#include "stdint.h"
#include "Generator_base.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class Generator_hifiGan: public Generator_base
{
public:
   Generator_hifiGan(float * modelData, int32_t & offset,int32_t isMS);
   ~Generator_hifiGan();
   MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};

#endif
