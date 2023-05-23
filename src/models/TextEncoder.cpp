#include "TextEncoder.h"
#include "tts_logger.h"
#include "attention_encoder.h"
#include "nn_conv1d.h"

using Eigen::Map;

typedef struct
{
    int32_t vocabSize_;
    int32_t embSize_;
    int32_t hiddenChannels_;
    MatrixXf emb_;
    attention_encoder * encoder_;
    nn_conv1d * proj_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

}TEXT_ENCODER_DATA_t;

TextEncoder::TextEncoder(float * modelData, int32_t & offset)
{
    TEXT_ENCODER_DATA_t * textEncoderData = new TEXT_ENCODER_DATA_t();
    if(NULL == textEncoderData)
    {
        tts_log(TTS_LOG_ERROR, "Text Encoder: Failed to allocate memory for internal data block\n");
        return;
    }
    memset(textEncoderData, 0, sizeof(TEXT_ENCODER_DATA_t));
    
    int32_t curOffset = offset;

    textEncoderData->hiddenChannels_ = (int32_t)modelData[curOffset++];
    textEncoderData->vocabSize_ = (int32_t)modelData[curOffset++];
    textEncoderData->embSize_ = (int32_t)modelData[curOffset++];

    textEncoderData->emb_ = Map<MatrixXf>(modelData+curOffset,
                                          textEncoderData->vocabSize_, 
                                          textEncoderData->embSize_);
    

    curOffset = curOffset + textEncoderData->vocabSize_*textEncoderData->embSize_;

    textEncoderData->encoder_ = new attention_encoder(modelData,curOffset);
    textEncoderData->proj_ = new nn_conv1d(modelData, curOffset);

    offset = curOffset;
    priv_ = (void *)textEncoderData;
}

MatrixXf TextEncoder::forward(int32_t *x, int32_t length, MatrixXf & m, MatrixXf & logs)
{
    TEXT_ENCODER_DATA_t * textEncoderData = (TEXT_ENCODER_DATA_t *)priv_;

    MatrixXf lengthM(1,1);
    lengthM(0,0) = (float)textEncoderData->hiddenChannels_;
    
    MatrixXf embOut = MatrixXf::Zero(length,textEncoderData->embSize_);

    for(int32_t i = 0; i<length; i++)
    {
        embOut.row(i) = textEncoderData->emb_.row(x[i]);
    }
    embOut = (embOut.array()*(lengthM.array().sqrt()(0,0))).matrix();

    MatrixXf XX = embOut;

    XX = textEncoderData->encoder_->forward(XX);
    MatrixXf stat = textEncoderData->proj_->forward(XX);

    m = stat.block(0,0,stat.rows(), stat.cols()/2); 
    logs = stat.block(0,stat.cols()/2,stat.rows(), stat.cols()/2);

    return XX;
}

TextEncoder::~TextEncoder()
{
    TEXT_ENCODER_DATA_t * textEncoderData = (TEXT_ENCODER_DATA_t *)priv_;

    delete textEncoderData->encoder_;
    delete textEncoderData->proj_;
    delete textEncoderData;
}

