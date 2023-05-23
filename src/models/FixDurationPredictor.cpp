#include "FixDurationPredictor.h"
#include "tts_logger.h"
#include "nn_conv1d.h"
#include "nn_layer_norm.h"
#include "nn_relu.h"

typedef struct
{
    int32_t isMS_;
    int32_t gin_channels_;
    nn_conv1d * conv_1_;
    nn_layer_norm * norm_1_;
    nn_conv1d * conv_2_;
    nn_layer_norm * norm_2_;
    nn_conv1d * proj_;
    nn_conv1d * cond_;
}DUR_PRED_t;

FixDurationPredictor::FixDurationPredictor(float * modelData, int32_t & offset, int32_t isMS)
{
    DUR_PRED_t * durPredData = new DUR_PRED_t();
    if(NULL == durPredData)
    {
        tts_log(TTS_LOG_ERROR, "Stoch Duration Preditor: Failed to allocate memory for internal data block\n");
        return;
    }
    memset(durPredData, 0, sizeof(DUR_PRED_t));

    int32_t curOffset = offset;

    durPredData->isMS_ = isMS;

    durPredData->conv_1_ = new nn_conv1d(modelData, curOffset);
    durPredData->norm_1_ = new nn_layer_norm(modelData, curOffset);
    durPredData->conv_2_ = new nn_conv1d(modelData, curOffset);
    durPredData->norm_2_ = new nn_layer_norm(modelData, curOffset);
    durPredData->proj_ = new nn_conv1d(modelData, curOffset);
    durPredData->cond_ = NULL;

    
    if(durPredData->isMS_ == 1)
    {
        durPredData->cond_ = new nn_conv1d(modelData, curOffset);
    }

    offset = curOffset;
    priv_ = (void*)durPredData;
}

FixDurationPredictor::~FixDurationPredictor()
{
    DUR_PRED_t * durPredData = (DUR_PRED_t *)priv_;

    delete durPredData->conv_1_;
    delete durPredData->norm_1_;
    delete durPredData->conv_2_;
    delete durPredData->norm_2_;
    delete durPredData->proj_;

    if(durPredData->isMS_ == 1)
    {
        delete durPredData->cond_;
    }

    delete durPredData;
}

void FixDurationPredictor::setMSSpk(int32_t isMS, int32_t ginChannels)
{
    DUR_PRED_t * durPredData = (DUR_PRED_t *)priv_;
    durPredData->isMS_ = isMS;
    durPredData->gin_channels_ = ginChannels;
}

MatrixXf FixDurationPredictor::forward(const MatrixXf & x, const MatrixXf & g,float noiseScale)
{
    DUR_PRED_t * durPredData = (DUR_PRED_t *)priv_;

    MatrixXf XX = x;

    if(durPredData->isMS_ == 1)
    {
        MatrixXf gg = durPredData->cond_->forward(g);
        XX = XX.rowwise() + gg.row(0);
    }
    
    XX = durPredData->conv_1_->forward(XX);
    XX = nn_relu(XX);
    XX = durPredData->norm_1_->forward(XX);
    XX = durPredData->conv_2_->forward(XX);
    XX = nn_relu(XX);
    XX = durPredData->norm_2_->forward(XX);
    XX = durPredData->proj_->forward(XX);

    return XX;
}


