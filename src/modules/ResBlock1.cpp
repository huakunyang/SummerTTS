#include "ResBlock1.h"
#include "nn_conv1d.h"
#include "tts_logger.h"
#include "nn_leaky_relu.h"

typedef struct
{
    int32_t blockNums_;
    nn_conv1d ** convs1_;
    nn_conv1d ** convs2_;
}RESBLOCK1_DATA_t;


ResBlock1::ResBlock1(float * modelData, int32_t & offset)
{
    RESBLOCK1_DATA_t * resBlockData = new RESBLOCK1_DATA_t();
    if(NULL == resBlockData)
    {
        tts_log(TTS_LOG_ERROR, "ResBlock1: Failed to allocate memory for internal data block\n");
        return;
    }

    memset(resBlockData, 0, sizeof(RESBLOCK1_DATA_t));

    int32_t curOffset = offset;

    resBlockData->blockNums_ = (int32_t)modelData[curOffset++];
    resBlockData->convs1_ = (nn_conv1d**)malloc(sizeof(nn_conv1d*)*resBlockData->blockNums_);
    for(int32_t i = 0; i<resBlockData->blockNums_; i++)
    {
        resBlockData->convs1_[i] = new nn_conv1d(modelData,curOffset);
    }

    resBlockData->convs2_ = (nn_conv1d**)malloc(sizeof(nn_conv1d*)*resBlockData->blockNums_);
    for(int32_t i = 0; i<resBlockData->blockNums_; i++)
    {
        resBlockData->convs2_[i] = new nn_conv1d(modelData,curOffset);
    }

    offset = curOffset;
    priv_ = (void *)resBlockData;
}

ResBlock1::~ResBlock1()
{
    RESBLOCK1_DATA_t * resBlockData =  (RESBLOCK1_DATA_t *)priv_;
    for(int32_t i = 0; i<resBlockData->blockNums_; i++)
    {
        delete resBlockData->convs1_[i];
        delete resBlockData->convs2_[i];
    }
    delete resBlockData;
}

MatrixXf ResBlock1::forward(const MatrixXf & x)
{
    RESBLOCK1_DATA_t * resBlockData =  (RESBLOCK1_DATA_t *)priv_;

    MatrixXf xx = x;
    for(int32_t i = 0; i<resBlockData->blockNums_; i++)
    {
        MatrixXf xt = nn_leaky_relu(xx,0.1);
        xt = resBlockData->convs1_[i]->forward(xt);
        xt = nn_leaky_relu(xt,0.1);
        xt = resBlockData->convs2_[i]->forward(xt);
        xx = xt + xx;
    }
    return xx;
}


