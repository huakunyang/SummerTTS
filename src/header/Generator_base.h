#ifndef _GENERATOR_BASE_H_
#define _GENERATOR_BASE_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class Generator_base
{
public:
    virtual ~Generator_base() = 0;
    virtual MatrixXf forward(const MatrixXf & x, const MatrixXf & g) = 0;

};

#endif
