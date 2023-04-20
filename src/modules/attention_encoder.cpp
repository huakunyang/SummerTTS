#include "attention_encoder.h"
#include "multi_head_attention.h"
#include "nn_layer_norm.h"
#include "tts_logger.h"
#include "ffn.h"

typedef struct
{
    int32_t numLayers_;
    multi_head_attention ** multiHeadAttnList;
    nn_layer_norm ** layerNormList1_;
    FFN ** ffnList_;
    nn_layer_norm ** layerNormList2_;

}ATTENTION_ENCODER_t;

attention_encoder::attention_encoder(float * modelData, int32_t & offset)
{
    ATTENTION_ENCODER_t * attenEncoderData = new ATTENTION_ENCODER_t();
    if(NULL == attenEncoderData)
    {
        tts_log(TTS_LOG_ERROR, "Module Attn Encoder: Failed to allocate memory for internal data block\n");
        return;
    }

    memset(attenEncoderData, 0, sizeof(ATTENTION_ENCODER_t));
    
    int32_t curOffset = offset;
    
    attenEncoderData->numLayers_ = (int32_t)modelData[curOffset++];

    attenEncoderData->multiHeadAttnList = (multi_head_attention** )(malloc(sizeof(multi_head_attention *)*attenEncoderData->numLayers_));
    attenEncoderData->layerNormList1_ = (nn_layer_norm**)(malloc(sizeof(nn_layer_norm*)*attenEncoderData->numLayers_));
    attenEncoderData->ffnList_ = (FFN**)(malloc(sizeof(FFN*)*attenEncoderData->numLayers_));
    attenEncoderData->layerNormList2_ = (nn_layer_norm**)(malloc(sizeof(nn_layer_norm*)*attenEncoderData->numLayers_));

    for(int32_t i = 0; i< attenEncoderData->numLayers_; i++)
    {
        attenEncoderData->multiHeadAttnList[i] = new multi_head_attention(modelData,curOffset);
    }

    for(int32_t i = 0; i< attenEncoderData->numLayers_; i++)
    {
        attenEncoderData->layerNormList1_[i] = new nn_layer_norm(modelData,curOffset);
    }

    for(int32_t i = 0; i< attenEncoderData->numLayers_; i++)
    {
        attenEncoderData->ffnList_[i] = new FFN(modelData,curOffset);
    }

    for(int32_t i = 0; i< attenEncoderData->numLayers_; i++)
    {
        attenEncoderData->layerNormList2_[i] = new nn_layer_norm(modelData,curOffset);
    }
    
    offset = curOffset;

    priv_ = (void *)attenEncoderData;
}

attention_encoder::~attention_encoder()
{
    ATTENTION_ENCODER_t * attenEncoderData = (ATTENTION_ENCODER_t *)priv_;

    for(int32_t i =0; i< attenEncoderData->numLayers_; i++)
    {
        delete attenEncoderData->multiHeadAttnList[i];
        delete attenEncoderData->layerNormList1_[i];
        delete attenEncoderData->ffnList_[i];
        delete attenEncoderData->layerNormList2_[i];
    }

    free((void *)attenEncoderData->multiHeadAttnList);
    delete attenEncoderData;
}

MatrixXf attention_encoder::forward(const MatrixXf & x)
{
    ATTENTION_ENCODER_t * attenEncoderData = (ATTENTION_ENCODER_t *)priv_;
    
    MatrixXf XX = x;
    for(int32_t i = 0; i<attenEncoderData->numLayers_; i++)
    {
        MatrixXf y = attenEncoderData->multiHeadAttnList[i]->forward(XX,XX);
        MatrixXf x1 = attenEncoderData->layerNormList1_[i]->forward(XX+y);
        y = attenEncoderData->ffnList_[i]->forward(x1);

        XX = attenEncoderData->layerNormList2_[i]->forward(x1+y);

    }

    return XX;
}
