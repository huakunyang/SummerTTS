#include "nn_sigmoid.h"

MatrixXf nn_sigmoid(const MatrixXf & x)
{
    MatrixXf ret = 1.0/(1.0+ (-x).array().exp());
    return ret;
}
