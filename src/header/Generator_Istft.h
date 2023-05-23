#ifndef _GENERATOR_ISTFT_H_
#define _GENERATOR_ISTFT_H_

#include "stdint.h"
#include <Eigen/Dense>
#include "Generator_base.h"

using Eigen::MatrixXf;

class Generator_Istft: public Generator_base
{
public:
   Generator_Istft(float * modelData, int32_t & offset,int32_t isMS);
   ~Generator_Istft();
   MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};

#endif
