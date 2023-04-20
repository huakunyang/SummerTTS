#ifndef _TTS_NN_CONV1D_TRANSPOSED_H_
#define _TTS_NN_CONV1D_TRANSPOSED_H_

#include <Eigen/Dense>

using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::Map;
using Eigen::ArrayXXf;
using Eigen::Array;
using Eigen::Dynamic;

int32_t parse_conv1d_transposed_parameter(float * modelData, int32_t & offset,
                                          int32_t & inCh, int32_t & outCh, int32_t & kSize,
                                          int32_t & padding, int32_t & dilation, 
                                          int32_t & hasBias,int32_t & stride,
                                          MatrixXf & weight, MatrixXf & bias);

class nn_conv1d_transposed
{
public:
    nn_conv1d_transposed(float * modelData, int32_t & offset,int32_t stride, int32_t padding);

    nn_conv1d_transposed(int32_t inCh, int32_t outCh, int32_t kSize, 
                         int32_t padding, int32_t dilation,
                         int32_t hasBias, int32_t stride,
                         MatrixXf & weight, MatrixXf & bias);

    int32_t get_in_channels_num();
    int32_t get_out_channels_num();

    MatrixXf forward(const MatrixXf & inputMat);
    ~nn_conv1d_transposed();
private:
    void * priv_;
    
};


#endif
