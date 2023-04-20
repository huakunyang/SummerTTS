#include "ffn.h"
#include "tts_logger.h"
#include "nn_conv1d.h"
#include "nn_relu.h"

typedef struct
{
    int32_t kSize_;
    nn_conv1d * conv1_;
    nn_conv1d * conv2_;

}FFN_DATA_t;

FFN::FFN(float * modelData, int32_t & offset)
{
    FFN_DATA_t * ffnData = new FFN_DATA_t();
    if(NULL == ffnData)
    {
        tts_log(TTS_LOG_ERROR, "FFN: Failed to allocate memory for internal data block\n");
        return;
    }

    memset(ffnData, 0, sizeof(FFN_DATA_t));

    int32_t curOffset = offset;
    
    ffnData->kSize_ = (int32_t)modelData[curOffset++];
    
    ffnData->conv1_ = new nn_conv1d(modelData, curOffset);
    ffnData->conv2_ = new nn_conv1d(modelData, curOffset);

    offset = curOffset;

    priv_ = (void *)ffnData;

}

FFN::~FFN()
{
    FFN_DATA_t * ffnData = (FFN_DATA_t *)priv_;
    delete ffnData->conv1_;
    delete ffnData->conv2_;

    delete ffnData;
}

MatrixXf same_padding(const MatrixXf & x, int32_t kSize)
{
    MatrixXf ret = x;
    if(kSize == 1)
    {
        return ret;
    }

    int32_t pad_l = floor((float)(kSize-1)/2);
    int32_t pad_r = floor((float)kSize/2);

    ret = MatrixXf::Zero(x.rows()+pad_l+pad_r, x.cols());
    ret.block(pad_l, 0, x.rows(), x.cols()) = x;

    return ret;
}

MatrixXf FFN::forward(const MatrixXf & x)
{
    FFN_DATA_t * ffnData = (FFN_DATA_t *)priv_;


    MatrixXf paddingX = same_padding(x, ffnData->kSize_);
    MatrixXf x_conv1 = ffnData->conv1_->forward(paddingX);

    MatrixXf relu_x = nn_relu(x_conv1);
    paddingX = same_padding(relu_x, ffnData->kSize_);
    MatrixXf x_conv2 = ffnData->conv2_->forward(paddingX);

    return x_conv2;
}
