#include "nn_leaky_relu.h"
#define SLOPE (1e-2)



MatrixXf nn_leaky_relu(const MatrixXf & x)
{
    return nn_leaky_relu(x, SLOPE);
}

MatrixXf nn_leaky_relu(const MatrixXf & x, float slope)
{
    MatrixXf result = x;

    for(int32_t i = 0; i< x.rows(); i++)
    {
        for(int j = 0; j< x.cols(); j++)
        {
            if(x(i,j) < 0)
            {
                result(i,j) = x(i,j)*slope;
            }
        }
    }

    return result;
}
