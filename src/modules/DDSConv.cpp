#include "DDSConv.h"
#include "nn_conv1d.h"
#include "nn_gelu.h"
#include "nn_layer_norm.h"
#include "tts_logger.h"

typedef struct
{
    int32_t layers_;
    int32_t kSize_;
    nn_conv1d ** convs_sep_;
    nn_conv1d ** convs_1_1_;
    nn_layer_norm ** norms_1_;
    nn_layer_norm ** norms_2_;
}DDS_CONV_DATA_t;

DDSConv::DDSConv(float * modelData, int32_t & offset)
{
    DDS_CONV_DATA_t * ddsConvData = new DDS_CONV_DATA_t();
    if(NULL == ddsConvData)
    {
        tts_log(TTS_LOG_ERROR, "DDS Conv: Failed to allocate memory for internal data block\n");
        return;
    }
    
    memset(ddsConvData,0,sizeof(DDS_CONV_DATA_t));
    int32_t curOffset = offset;
    
    ddsConvData->layers_ = (int32_t)modelData[curOffset++];
    ddsConvData->kSize_ = (int32_t)modelData[curOffset++];

    ddsConvData->convs_sep_ = (nn_conv1d **)(malloc(sizeof(nn_conv1d *) * ddsConvData->layers_));

    int32_t convs_sep_dilation = 1;
    int32_t convs_sep_padding = 0;
    for(int32_t i = 0; i< ddsConvData->layers_; i++)
    {
        convs_sep_padding = floor((float)(ddsConvData->kSize_ * convs_sep_dilation - convs_sep_dilation)/2.0);
        ddsConvData->convs_sep_[i] = new nn_conv1d(modelData, curOffset, convs_sep_padding, convs_sep_dilation,1);
        convs_sep_dilation = convs_sep_dilation * ddsConvData->kSize_;
    }

    ddsConvData->convs_1_1_ = (nn_conv1d **)(malloc(sizeof(nn_conv1d *) * ddsConvData->layers_));
    for(int32_t i = 0; i< ddsConvData->layers_; i++)
    {
        ddsConvData->convs_1_1_[i] = new nn_conv1d(modelData, curOffset);
    }

    ddsConvData->norms_1_ = (nn_layer_norm **)(malloc(sizeof(nn_layer_norm *) * ddsConvData->layers_));
    for(int32_t i = 0; i< ddsConvData->layers_; i++)
    {
        ddsConvData->norms_1_[i] = new nn_layer_norm(modelData, curOffset);
    }

    ddsConvData->norms_2_ = (nn_layer_norm **)(malloc(sizeof(nn_layer_norm *) * ddsConvData->layers_));
    for(int32_t i = 0; i< ddsConvData->layers_; i++)
    {
        ddsConvData->norms_2_[i] = new nn_layer_norm(modelData, curOffset);
    }

    offset = curOffset;
    priv_ = (void *)ddsConvData;
}

DDSConv::~DDSConv()
{
    DDS_CONV_DATA_t * ddsConvData = (DDS_CONV_DATA_t*)priv_;
    
    for(int32_t i = 0; i<ddsConvData->layers_; i++)
    {
        delete ddsConvData->convs_sep_[i];
        delete ddsConvData->convs_1_1_[i];
        delete ddsConvData->norms_1_[i];
        delete ddsConvData->norms_2_[i];
    }

    delete ddsConvData->convs_sep_;
    delete ddsConvData->convs_1_1_;
    delete ddsConvData->norms_1_;
    delete ddsConvData->norms_2_;
    delete ddsConvData;
}

MatrixXf DDSConv::forward(const MatrixXf & x, const MatrixXf & g, int32_t hasG)
{
    DDS_CONV_DATA_t * ddsConvData = (DDS_CONV_DATA_t*)priv_;
    
    MatrixXf xx;
    if(hasG == 1)
    {
        xx = x + g; 
    }
    else
    {
        xx = x;
    }
    for(int32_t i = 0; i<ddsConvData->layers_; i++)
    {
        MatrixXf y = ddsConvData->convs_sep_[i]->forward(xx);

        y = ddsConvData->norms_1_[i]->forward(y);

        y = nn_gelu(y);
        y = ddsConvData->convs_1_1_[i]->forward(y);
        y = ddsConvData->norms_2_[i]->forward(y);
        y = nn_gelu(y);
        xx = xx + y;
    }

    return xx;
}
