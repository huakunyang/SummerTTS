#include "StochasticDurationPredictor.h"
#include "tts_logger.h"
#include "ElementwiseAffine.h"
#include "ConvFlow.h"
#include "nn_conv1d.h"
#include "nn_flip.h"
#include "random_gen.h"
#include "DDSConv.h"

typedef struct
{
    int32_t isMS_;
    int32_t gin_channels_;
    int32_t nFlows_;
    ElementwiseAffine * FlowElementWiseAffine_;
    ConvFlow ** FlowConvFlowList_;
    nn_conv1d * post_pre_;
    nn_conv1d * post_proj_;
    DDSConv * post_convs_;
    ElementwiseAffine * PostflowElementWiseAffine_;
    ConvFlow ** PostflowConvFlowList_;
    nn_conv1d * pre_;
    nn_conv1d * proj_;
    DDSConv * convs_;
    nn_conv1d * cond_;
}STOCH_DUR_PRED_t;

StochasticDurationPredictor::StochasticDurationPredictor(float * modelData, int32_t & offset, int32_t isMS)
{
    STOCH_DUR_PRED_t * stochDurPredData = new STOCH_DUR_PRED_t();
    if(NULL == stochDurPredData)
    {
        tts_log(TTS_LOG_ERROR, "Stoch Duration Preditor: Failed to allocate memory for internal data block\n");
        return;
    }
    memset(stochDurPredData, 0, sizeof(STOCH_DUR_PRED_t));

    int32_t curOffset = offset;

    stochDurPredData->isMS_ = isMS;
    stochDurPredData->nFlows_ = (int32_t)modelData[curOffset++];
    stochDurPredData->FlowElementWiseAffine_ = new ElementwiseAffine(modelData, curOffset, 2);
    
    stochDurPredData->FlowConvFlowList_ = (ConvFlow **)(malloc(sizeof(ConvFlow *)*stochDurPredData->nFlows_));
    for(int32_t i = 0; i<stochDurPredData->nFlows_; i++)
    {
        stochDurPredData->FlowConvFlowList_[i] = new ConvFlow(modelData, curOffset);
    }

    stochDurPredData->post_pre_ = new nn_conv1d(modelData, curOffset);

    stochDurPredData->post_proj_ = new nn_conv1d(modelData, curOffset);
    stochDurPredData->post_convs_ = new DDSConv(modelData, curOffset);
    stochDurPredData->PostflowElementWiseAffine_ = new ElementwiseAffine(modelData, curOffset, 2);

    stochDurPredData->PostflowConvFlowList_ = (ConvFlow**)(malloc(sizeof(ConvFlow *)*4));
    for(int32_t i = 0; i<4; i++)
    {
        stochDurPredData->PostflowConvFlowList_[i] = new ConvFlow(modelData, curOffset);
    }

    stochDurPredData->pre_ = new nn_conv1d(modelData, curOffset);
    stochDurPredData->proj_ = new nn_conv1d(modelData, curOffset);
    stochDurPredData->convs_ = new DDSConv(modelData, curOffset);
    stochDurPredData->cond_ = NULL;

    if(stochDurPredData->isMS_ == 1)
    {
        stochDurPredData->cond_ = new nn_conv1d(modelData, curOffset);
    }

    offset = curOffset;
    priv_ = (void*)stochDurPredData;
}

StochasticDurationPredictor::~StochasticDurationPredictor()
{
    STOCH_DUR_PRED_t * stochDurPredData = (STOCH_DUR_PRED_t *)priv_;
    delete stochDurPredData->FlowElementWiseAffine_;
    
    for(int32_t i = 0; i< stochDurPredData->nFlows_; i++)
    {
        delete stochDurPredData->FlowConvFlowList_[i];
    }
    free(stochDurPredData->FlowConvFlowList_);

    delete stochDurPredData->post_pre_;
    delete stochDurPredData->post_proj_;
    delete stochDurPredData->post_convs_;
    delete stochDurPredData->PostflowElementWiseAffine_;

    for(int32_t i = 0; i<4; i++)
    {
        delete stochDurPredData->PostflowConvFlowList_[i];
    }
    free(stochDurPredData->PostflowConvFlowList_);

    if(stochDurPredData->isMS_ == 1)
    {
        delete stochDurPredData->cond_;
    }

    delete stochDurPredData->pre_;
    delete stochDurPredData->proj_;
    delete stochDurPredData->convs_;

    delete stochDurPredData;
}

void StochasticDurationPredictor::setMSSpk(int32_t isMS, int32_t ginChannels)
{
    STOCH_DUR_PRED_t * stochDurPredData = (STOCH_DUR_PRED_t *)priv_;
    stochDurPredData->isMS_ = isMS;
    stochDurPredData->gin_channels_ = ginChannels;
}

MatrixXf StochasticDurationPredictor::forward(const MatrixXf & x, const MatrixXf & g,float noiseScale)
{
    STOCH_DUR_PRED_t * stochDurPredData = (STOCH_DUR_PRED_t *)priv_;

    MatrixXf XX = stochDurPredData->pre_->forward(x); 

    if(stochDurPredData->isMS_ == 1)
    {
        MatrixXf gg = stochDurPredData->cond_->forward(g);
        XX = XX.rowwise() + gg.row(0);
    }
    
    MatrixXf dummy;
    XX = stochDurPredData->convs_->forward(XX,dummy,0);
    XX = stochDurPredData->proj_->forward(XX);

    MatrixXf z = rand_gen(2,XX.rows(),0.0,1.0)*noiseScale;

    MatrixXf flapZ = nn_flip(z,0);
    MatrixXf flapZ_T = flapZ;

    for(int32_t i = stochDurPredData->nFlows_-1; i>0; i--)
    {
        flapZ = stochDurPredData->FlowConvFlowList_[i]->forward(flapZ_T, XX);
        flapZ = nn_flip(flapZ,1);
        flapZ_T = flapZ.transpose();
    }
    
    MatrixXf affinedM = stochDurPredData->FlowElementWiseAffine_->forward(flapZ);
    MatrixXf ret = affinedM.block(0,0,(int32_t)affinedM.rows(),affinedM.cols()/2);

    return ret;
}


