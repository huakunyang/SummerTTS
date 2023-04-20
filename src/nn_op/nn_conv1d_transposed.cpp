#include "nn_conv1d_transposed.h"
#include "tts_logger.h"

typedef struct
{
    int32_t outCh_;
    int32_t inCh_;
    int32_t kSize_;
    int32_t padding_;
    int32_t dilation_;
    int32_t hasBias_;
    int32_t stride_;
    

    MatrixXf w_;
    MatrixXf b_;
    
    int32_t print_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}NN_CONV1D_TRANSPOSED_DATA_t;


int32_t parse_conv1d_transposed_parameter(float * modelData, int32_t & offset,
                                          int32_t & inCh, int32_t & outCh, int32_t & kSize,
                                          int32_t & padding, int32_t & dilation, 
                                          int32_t & hasBias,int32_t & stride,
                                          MatrixXf & weight, MatrixXf & bias)
{

    int32_t curOffset = offset;

    outCh = (int32_t)modelData[curOffset++];
    inCh = (int32_t)modelData[curOffset++];
    kSize = (int32_t)modelData[curOffset++];
    padding = (int32_t)modelData[curOffset++];
    dilation = (int32_t)modelData[curOffset++];
    hasBias = (int32_t)modelData[curOffset++];
    stride = (int32_t)modelData[curOffset++];

    weight = Map<MatrixXf>(modelData+curOffset, inCh, kSize*outCh);
    curOffset = curOffset + inCh * kSize * outCh;

    if(1 == hasBias)
    {
        bias = Map<MatrixXf>(modelData+curOffset,1, outCh);
        curOffset = curOffset + 1*outCh;
    }

    offset = curOffset;

    return 1;
}

nn_conv1d_transposed::nn_conv1d_transposed(float * modelData, int32_t & offset, int32_t stride, int32_t padding)
{

    NN_CONV1D_TRANSPOSED_DATA_t * nn1dConvTransposedData = new NN_CONV1D_TRANSPOSED_DATA_t();
    if(NULL == nn1dConvTransposedData)
    {
        tts_log(TTS_LOG_ERROR,"NN CONV1D TRANSPOSED: failed to allocate memory for internal block\n");
    }
    
    int32_t curOffset = offset;
    parse_conv1d_transposed_parameter(modelData, curOffset, nn1dConvTransposedData->inCh_, 
                                      nn1dConvTransposedData->outCh_,
                                      nn1dConvTransposedData->kSize_, nn1dConvTransposedData->padding_,
                                      nn1dConvTransposedData->dilation_, nn1dConvTransposedData->hasBias_,
                                      nn1dConvTransposedData->stride_,
                                      nn1dConvTransposedData->w_, nn1dConvTransposedData->b_);

    nn1dConvTransposedData->stride_ = stride;
    nn1dConvTransposedData->padding_ = padding;

    offset = curOffset;
    priv_ = (void*)nn1dConvTransposedData;
}

nn_conv1d_transposed::nn_conv1d_transposed(int32_t inCh, int32_t outCh, int32_t kSize, 
                                           int32_t padding, int32_t dilation,
                                           int32_t hasBias, int32_t stride,
                                           MatrixXf & weight, MatrixXf & bias)
{

    NN_CONV1D_TRANSPOSED_DATA_t * nn1dConvTransposedData = new NN_CONV1D_TRANSPOSED_DATA_t();
    if(NULL == nn1dConvTransposedData)
    {
        tts_log(TTS_LOG_ERROR,"NN CONV1D: failed to allocate memory for internal block\n");
        return;
    }

    nn1dConvTransposedData->inCh_ = inCh;
    nn1dConvTransposedData->outCh_ = outCh;
    nn1dConvTransposedData->kSize_ = kSize;
    nn1dConvTransposedData->padding_ = padding;
    nn1dConvTransposedData->dilation_ = dilation;
    nn1dConvTransposedData->hasBias_ = hasBias;
    nn1dConvTransposedData->stride_ = stride;
    nn1dConvTransposedData->w_ = weight;
    nn1dConvTransposedData->b_ = bias;

    priv_ = (void*)nn1dConvTransposedData;

}

MatrixXf  nn_conv1d_transposed::forward(const MatrixXf & inputMat)
{
    NN_CONV1D_TRANSPOSED_DATA_t * nn1dConvTransposedData = (NN_CONV1D_TRANSPOSED_DATA_t *)priv_;

    int32_t inLen = inputMat.rows();
    int32_t outLen = (inLen-1)*nn1dConvTransposedData->stride_-
                     2*nn1dConvTransposedData->padding_+
                     nn1dConvTransposedData->dilation_*(nn1dConvTransposedData->kSize_-1)+1;

    MatrixXf result = MatrixXf::Zero(nn1dConvTransposedData->outCh_,outLen+2*(nn1dConvTransposedData->padding_)+nn1dConvTransposedData->kSize_);
    MatrixXf w = nn1dConvTransposedData->w_; 

    MatrixXf outMat = inputMat * w;

    int outIndex = 0;
    for(int32_t i = 0; i<inLen; i++)
    {
        MatrixXf outRow = outMat.row(i).transpose();

        MatrixXf outRowReshaped = outRow.reshaped(nn1dConvTransposedData->kSize_,nn1dConvTransposedData->outCh_);
        MatrixXf outRowT = outRowReshaped.transpose();

        result.block(0,outIndex,nn1dConvTransposedData->outCh_,nn1dConvTransposedData->kSize_)
        = result.block(0,outIndex,nn1dConvTransposedData->outCh_,nn1dConvTransposedData->kSize_) + outRowT;

        outIndex = outIndex + nn1dConvTransposedData->stride_;

        if((outIndex + (int32_t)(nn1dConvTransposedData->kSize_))>= result.cols())
        {
            break;
        }
    }

    MatrixXf resultT = result.transpose();

    if(1 == nn1dConvTransposedData->hasBias_)
    {
        resultT = resultT.rowwise() + nn1dConvTransposedData->b_.row(0);
    }

    MatrixXf resultTBlocked = resultT.block(nn1dConvTransposedData->padding_,0,
                                            outLen,nn1dConvTransposedData->outCh_);

    return resultTBlocked;
}

int32_t nn_conv1d_transposed::get_in_channels_num()
{
    NN_CONV1D_TRANSPOSED_DATA_t * nn1dConvTransposedData = (NN_CONV1D_TRANSPOSED_DATA_t *)priv_;
    return nn1dConvTransposedData->inCh_; 
}

int32_t nn_conv1d_transposed::get_out_channels_num()
{
    NN_CONV1D_TRANSPOSED_DATA_t * nn1dConvTransposedData = (NN_CONV1D_TRANSPOSED_DATA_t *)priv_;
    return nn1dConvTransposedData->outCh_; 
}

nn_conv1d_transposed::~nn_conv1d_transposed()
{
    NN_CONV1D_TRANSPOSED_DATA_t * nn1dConvTransposedData = (NN_CONV1D_TRANSPOSED_DATA_t *)priv_;
    delete nn1dConvTransposedData;
}
