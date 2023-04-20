#include "nn_layer_norm.h"
#include "tts_logger.h"

using Eigen::Map;
using Eigen::ArrayXf;

#define eps_norm (1e-05)

typedef struct
{
    int32_t size_;
    MatrixXf gamma_;
    MatrixXf beta_;

}NN_LAYER_NORM_t;

void parse_layernorm_parameter(float * modelData, int32_t & offset, int32_t & size,
                               MatrixXf & gamma, MatrixXf & beta)
{
    int32_t curOffset = offset;
    size = (int32_t)modelData[curOffset++];

    gamma = Map<MatrixXf>(modelData+curOffset,1,size);
    curOffset = curOffset + size;

    beta = Map<MatrixXf>(modelData+curOffset,1,size);

    curOffset = curOffset + size;
    
    offset = curOffset;
}

nn_layer_norm::nn_layer_norm(float * modelData, int32_t & offset)
{
    NN_LAYER_NORM_t * nnLayerNormData = new NN_LAYER_NORM_t();
    if(NULL == nnLayerNormData)
    {
        tts_log(TTS_LOG_ERROR,"NN LayerNorm: failed to allocate memory for internal block\n");
    }
    
    int32_t curOffset = offset;
    parse_layernorm_parameter(modelData, curOffset, nnLayerNormData->size_,
                              nnLayerNormData->gamma_,nnLayerNormData->beta_);

    offset = curOffset;

    priv_ = (void *)nnLayerNormData;
}

nn_layer_norm::nn_layer_norm(int32_t size, const MatrixXf & gamma, const MatrixXf & beta)
{
    NN_LAYER_NORM_t * nnLayerNormData = new NN_LAYER_NORM_t();
    if(NULL == nnLayerNormData)
    {
        tts_log(TTS_LOG_ERROR,"NN LayerNorm: failed to allocate memory for internal block\n");
    }
    
    nnLayerNormData->size_ = size;
    nnLayerNormData->gamma_ = gamma;
    nnLayerNormData->beta_ = beta;
    priv_ = (void *)nnLayerNormData;
}


MatrixXf nn_layer_norm::forward(const MatrixXf & x)
{
    NN_LAYER_NORM_t * nnLayerNormData = (NN_LAYER_NORM_t*)priv_;
    MatrixXf result = MatrixXf::Zero(x.rows(), x.cols());
    
    for(int32_t i = 0; i<x.rows(); i++)
    {
        MatrixXf m = x.row(i);
        Eigen::MatrixXf mean = m.rowwise().mean();
        float mean_ = mean(0, 0);
        Eigen::MatrixXf sqsum = (m * m.transpose()).rowwise().sum();
        float sqsum_ = sqsum(0, 0);
        float scale = 1. /(float)x.cols();
        float variance_ = sqsum_ * scale - mean_ * mean_;

        result.row(i) =  (((m.array() - mean_)/sqrt(variance_ + eps_norm))*
                          nnLayerNormData->gamma_.array()+
                          nnLayerNormData->beta_.array()).matrix();
    }

    return result;
}

