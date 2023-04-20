#include "nn_relu.h"

MatrixXf nn_relu(const MatrixXf & x)
{
    MatrixXf ret = x;
    for(int32_t i = 0; i< x.rows(); i++)
    {
        for(int32_t j=0; j< x.cols(); j++)
        {
            if(ret(i,j) < 0)
            {
                ret(i,j) = 0;
            }
        }
    }
    return ret;
}
