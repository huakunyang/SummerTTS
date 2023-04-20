#include "nn_cumsum.h"

MatrixXf nn_cumsum(const MatrixXf & x, int32_t dim)
{
    MatrixXf result = MatrixXf::Zero(x.rows(), x.cols());

    if(dim == 0)
    {
        result.row(0) = x.row(0);
        for(int32_t i = 1; i< x.rows(); i++)
        {
            result.row(i) = result.row(i-1)+x.row(i);
        }
    }
    else
    {
        result.col(0) = x.col(0);
        for(int32_t i = 1; i< x.cols(); i++)
        {
            result.col(i) = result.col(i-1)+x.col(i);
        }

    }
    
    return result;
}
