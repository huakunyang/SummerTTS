#ifndef _GENERATOR_MBB_H_
#define _GENERATOR_MBB_H_

#include "stdint.h"
#include <Eigen/Dense>
#include "Generator_base.h"


class Generator_MBB: public Generator_base
{
public:
   Generator_MBB(float * modelData, int32_t & offset,int32_t isMS);
   ~Generator_MBB();
   MatrixXf forward(const MatrixXf & x, const MatrixXf & g);

private:
    void * priv_;

};

#endif
