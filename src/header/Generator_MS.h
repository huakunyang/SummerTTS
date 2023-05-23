#ifndef _GENERATOR_MS_H_
#define _GENERATOR_MS_H_

#include "stdint.h"
#include "Generator_base.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class Generator_MS: public Generator_base
{
public:
   Generator_MS(float * modelData, int32_t & offset,int32_t isMS);
   ~Generator_MS();
   MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};

#endif
