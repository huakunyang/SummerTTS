#ifndef _ELEMENT_WISE_AFFINE_H_
#define _ELEMENT_WISE_AFFINE_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class ElementwiseAffine
{
public:
    ElementwiseAffine(float * modelData, int32_t & offset, int32_t channels);
    ~ElementwiseAffine();
    MatrixXf forward(const MatrixXf & x);

private:
    void * priv_;

};

#endif
