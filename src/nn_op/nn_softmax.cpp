#include "nn_softmax.h"

using Eigen::ArrayXf;

MatrixXf nn_softmax(const MatrixXf & mat, int32_t dim)
{
    MatrixXf expMat = mat.array().exp();
    MatrixXf ret = MatrixXf::Zero(expMat.rows(),expMat.cols());

    if(dim  == 0)
    {
        MatrixXf sumMat = expMat.rowwise().sum();
        for(int32_t i = 0; i< expMat.rows(); i++)
        {
            ret.row(i) = expMat.row(i)/sumMat(i);
        }
    }
    else
    {
        MatrixXf sumMat = expMat.colwise().sum();
        for(int32_t i = 0; i< expMat.cols(); i++)
        {
            ret.col(i) = expMat.col(i)/sumMat(i);
        }
    }

    return ret;
}
