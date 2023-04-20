#include "nn_softplus.h"

MatrixXf nn_softplus(const MatrixXf & x)
{
    MatrixXf ret = (x.array().exp()+1).log();

    return ret;
}
