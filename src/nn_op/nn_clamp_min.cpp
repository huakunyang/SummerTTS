#include "nn_clamp_min.h"

MatrixXf nn_clamp_min(const MatrixXf & x,float min)
{
    MatrixXf ret = x;
    for(int32_t i = 0; i<ret.rows(); i++)
    {
        for(int32_t j = 0; j<ret.cols(); j++)
        {
            if(ret(i,j) < min)
            {
                ret(i,j) = min;
            }
        }
    }

    return ret;
}
