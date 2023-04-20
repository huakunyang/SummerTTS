#ifndef _GENERATOR_H_
#define _GENERATOR_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class Generator
{
public:
   Generator(float * modelData, int32_t & offset,int32_t isMS);
   ~Generator();
   MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};

#endif
