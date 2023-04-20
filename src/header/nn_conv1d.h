#ifndef _TTS_NN_CONV1D_H_
#define _TTS_NN_CONV1D_H_

#include <Eigen/Dense>

using Eigen::MatrixXf;

int32_t parse_conv1d_parameter(float * modelData, int32_t & offset,
                               int32_t & inCh, int32_t & outCh, int32_t & kSize,
                               int32_t & padding, int32_t & dilation_, int32_t & hasBias,
                               MatrixXf & weight, MatrixXf & bias);

class nn_conv1d
{
public:
    nn_conv1d(float * modelData, int32_t & offset);
    nn_conv1d(float * modelData, int32_t & offset,int32_t padding, int32_t dilation, int sep);
    nn_conv1d(int32_t inCh, int32_t outCh, int32_t kSize, 
              int32_t padding, int32_t dilation, int32_t hasBias,  MatrixXf & weight, MatrixXf & bias);

    int32_t get_in_channels_num();
    int32_t get_out_channels_num();
    MatrixXf forward(MatrixXf inputMat);
    void print_p();
    ~nn_conv1d();
private:
    void * priv_;
    
};


#endif
