#include "nn_flip.h"

MatrixXf nn_flip(const MatrixXf & x, int32_t dim)
{
    MatrixXf ret;
    if(dim == 1)
    {
        ret = x.rowwise().reverse();
    }
    else
    {
        ret = x.colwise().reverse();
    }
    return ret;
}
