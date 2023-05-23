#include "multi_head_attention.h"
#include "tts_logger.h"
#include "nn_conv1d.h"
#include "nn_softmax.h"
#include <vector>

using Eigen::Map;

typedef struct
{
    int32_t channels_;
    int32_t outChannes_;
    int32_t nHeads_;
    int32_t winSize_;
    int32_t kChannels_;
    MatrixXf embRelK_;
    MatrixXf embRelV_;
    MatrixXf attn_;
    nn_conv1d * q_;
    nn_conv1d * k_;
    nn_conv1d * v_;
    nn_conv1d * o_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}MODULE_MULTI_HEAD_ATTN_DATA_t;

multi_head_attention::multi_head_attention(float * modelData, int32_t & offset)
{
    MODULE_MULTI_HEAD_ATTN_DATA_t * multiAttnData = new MODULE_MULTI_HEAD_ATTN_DATA_t();
    
    if(NULL == multiAttnData)
    {
        tts_log(TTS_LOG_ERROR, "Module Multi Attn: Failed to allocate memory for internal data block\n");
        return;
    }

    memset(multiAttnData,0,sizeof(MODULE_MULTI_HEAD_ATTN_DATA_t));
    
    int32_t curOffset = offset;

    multiAttnData->channels_ = (int32_t)modelData[curOffset++];
    multiAttnData->outChannes_ = (int32_t)modelData[curOffset++]; 
    multiAttnData->nHeads_ = (int32_t)modelData[curOffset++];
    multiAttnData->winSize_ = (int32_t)modelData[curOffset++];

    multiAttnData->kChannels_ = floor((float)multiAttnData->channels_/(float)multiAttnData->nHeads_);

    if(0 != multiAttnData->winSize_)
    {
        int32_t paramX = (int32_t)modelData[curOffset++];
        int32_t paramY = (int32_t)modelData[curOffset++];

        multiAttnData->embRelK_ = Map<MatrixXf>(modelData+curOffset, paramX,paramY);
        curOffset = curOffset + paramX*paramY;

        paramX = (int32_t)modelData[curOffset++];
        paramY = (int32_t)modelData[curOffset++];

        multiAttnData->embRelV_ = Map<MatrixXf>(modelData+curOffset, paramX,paramY);
        curOffset = curOffset + paramX*paramY;
    }

    multiAttnData->q_ = new nn_conv1d(modelData, curOffset);

    if(NULL == multiAttnData->q_)
    {
        tts_log(TTS_LOG_ERROR, "Module Multi Attn: Failed to init q_\n");
        delete multiAttnData;
        return;
    }

    multiAttnData->k_ = new nn_conv1d(modelData, curOffset);
    if(NULL == multiAttnData->k_)
    {
        tts_log(TTS_LOG_ERROR, "Module Multi Attn: Failed to init k_\n");
        delete multiAttnData->q_;
        delete multiAttnData;
        return;
    }

    multiAttnData->v_ = new nn_conv1d(modelData, curOffset);
    if(NULL == multiAttnData->v_)
    {
        tts_log(TTS_LOG_ERROR, "Module Multi Attn: Failed to init v_\n");
        delete multiAttnData->k_;
        delete multiAttnData->q_;
        delete multiAttnData;
        return;
    }

    multiAttnData->o_ = new nn_conv1d(modelData, curOffset);
    if(NULL == multiAttnData->o_)
    {
        tts_log(TTS_LOG_ERROR, "Module Multi Attn: Failed to init o_\n");
        delete multiAttnData->v_;
        delete multiAttnData->k_;
        delete multiAttnData->q_;
        delete multiAttnData;
        return;
    }

    offset = curOffset;

    priv_ = (void *)multiAttnData;
}

MatrixXf multi_head_attention::forward(MatrixXf x, MatrixXf c)
{
    MODULE_MULTI_HEAD_ATTN_DATA_t * multiAttnData = (MODULE_MULTI_HEAD_ATTN_DATA_t* )priv_;
    
    MatrixXf q = multiAttnData->q_->forward(x);
    MatrixXf k = multiAttnData->k_->forward(c);
    MatrixXf v = multiAttnData->v_->forward(c);

    MatrixXf attRet;
    attRet = attention(q,k,v);
    
    MatrixXf result;
    result = multiAttnData->o_->forward(attRet.transpose());

    return result;
}

static int32_t max(int32_t a, int32_t b)
{
    int32_t retVal = b;
    if(a>b)
    {
        retVal = a;
    }
    return retVal;
}

MatrixXf multi_head_attention::get_relative_embeddings(const MatrixXf & relativeEmbeddings, int32_t length)
{
    MODULE_MULTI_HEAD_ATTN_DATA_t * multiAttnData = (MODULE_MULTI_HEAD_ATTN_DATA_t* )priv_;

    int32_t maxRelativePosition = 2 * multiAttnData->winSize_ + 1;
    int32_t padLength = max(length - (multiAttnData->winSize_ + 1), 0);

    int32_t sliceStartPosition = max((multiAttnData->winSize_ + 1) - length, 0);
    int32_t sliceEndPosition = sliceStartPosition + 2 * length - 1;

    MatrixXf paddedRelativeEmbeddings = relativeEmbeddings;
    if (padLength > 0)
    {
        paddedRelativeEmbeddings = MatrixXf::Zero(relativeEmbeddings.rows()+2*padLength,relativeEmbeddings.cols());
        paddedRelativeEmbeddings.block(padLength,0,relativeEmbeddings.rows(),relativeEmbeddings.cols()) = relativeEmbeddings;
    }

    return paddedRelativeEmbeddings;
}

std::vector<MatrixXf> multi_head_attention::relative_position_to_absolute_position(const std::vector<MatrixXf> & x)
{
    MODULE_MULTI_HEAD_ATTN_DATA_t * multiAttnData = (MODULE_MULTI_HEAD_ATTN_DATA_t* )priv_;
    int32_t length = x[0].rows();
    int32_t heads = multiAttnData->nHeads_;

    std::vector<MatrixXf> paddedX;
    for(int32_t i = 0; i<heads; i++)
    {
        MatrixXf paddedMat = MatrixXf::Zero(x[i].rows(),x[i].cols()+1);
        paddedMat.block(0,0,x[i].rows(),x[i].cols()) = x[i];

        MatrixXf paddedMat2 = paddedMat.transpose().reshaped(1,paddedMat.rows()*paddedMat.cols());
        MatrixXf paddedMat3 = MatrixXf::Zero(paddedMat2.rows(),paddedMat2.cols()+length-1);
        paddedMat3.block(0,0,paddedMat2.rows(),paddedMat2.cols()) = paddedMat2;

        MatrixXf  paddedMat4 = paddedMat3.reshaped(2*length-1, length + 1).transpose();
        MatrixXf  paddedMat5 = paddedMat4.block(0, length-1,length,2*length-1-(length-1)); 
        paddedX.push_back(paddedMat5);
    }
    return paddedX;
}

std::vector<MatrixXf> multi_head_attention::absolute_position_to_relative_position(const std::vector<MatrixXf> & x)
{
    MODULE_MULTI_HEAD_ATTN_DATA_t * multiAttnData = (MODULE_MULTI_HEAD_ATTN_DATA_t* )priv_;
    int32_t length = x[0].rows();
    int32_t heads = multiAttnData->nHeads_;

    std::vector<MatrixXf> paddedX;
    for(int32_t i = 0; i<heads; i++)
    {
        MatrixXf paddedMat = MatrixXf::Zero(x[i].rows(),x[i].cols()+length-1);
        paddedMat.block(0,0,x[i].rows(),x[i].cols()) = x[i];

        MatrixXf paddedMat2 = paddedMat.transpose().reshaped(1,paddedMat.rows()*paddedMat.cols());

        MatrixXf paddedMat3 = MatrixXf::Zero(paddedMat2.rows(),paddedMat2.cols()+length);
        paddedMat3.block(0,length,paddedMat2.rows(),paddedMat2.cols()) = paddedMat2;

        MatrixXf  paddedMat4 = paddedMat3.reshaped(2*length, length).transpose();

        MatrixXf  paddedMat5 = paddedMat4.block(0, 1,length,2*length-1); 
        paddedX.push_back(paddedMat5);
    }
    return paddedX;
}

MatrixXf multi_head_attention::attention(const MatrixXf & query, const MatrixXf & key, const MatrixXf & value)
{
    MODULE_MULTI_HEAD_ATTN_DATA_t * multiAttnData = (MODULE_MULTI_HEAD_ATTN_DATA_t* )priv_;

    MatrixXf keyT = key.transpose();
    MatrixXf valueT = value.transpose();

    int32_t b = 1;

    int32_t d = keyT.rows();
    int32_t t_s = keyT.cols();
    
    std::vector<MatrixXf> qMatVec;
    int32_t queryColSplit = (int32_t)(query.cols()/multiAttnData->nHeads_);

    MatrixXf matKChannels(1,1);
    matKChannels(0,0) = (float)(multiAttnData->kChannels_);
    
    for(int32_t i = 0; i<multiAttnData->nHeads_; i++)
    {
        MatrixXf qSqrt = ((query.block(0,i*queryColSplit,query.rows(),queryColSplit)).array() /
                         matKChannels.array().sqrt()(0,0)).matrix();
                         
        qMatVec.push_back(qSqrt);
    }

    std::vector<MatrixXf> kMatVec;
    int32_t keyRowSplit = (int32_t)(keyT.rows()/multiAttnData->nHeads_);

    for(int32_t i = 0; i<multiAttnData->nHeads_; i++)
    {
        kMatVec.push_back(keyT.block(i*keyRowSplit,0,keyRowSplit,keyT.cols()));
    }
    
    std::vector<MatrixXf> scoreMatVec;
    for(int32_t i = 0; i<multiAttnData->nHeads_; i++)
    {
        scoreMatVec.push_back(qMatVec[i] * kMatVec[i]);
    }
   
    if(multiAttnData->winSize_ > 0)
    {
        MatrixXf keyRelativeEmbeddings = get_relative_embeddings(multiAttnData->embRelK_,t_s);

        std::vector<MatrixXf> relLogitsVec;
        for(int32_t i = 0; i<multiAttnData->nHeads_; i++)
        {
            relLogitsVec.push_back(qMatVec[i]*keyRelativeEmbeddings.transpose());
        }

        std::vector<MatrixXf> scores_local = relative_position_to_absolute_position(relLogitsVec); 
   
        for(int32_t i =0; i<multiAttnData->nHeads_; i++)
        {
            scoreMatVec[i] = scoreMatVec[i] + scores_local[i];
        }
    }

    std::vector<MatrixXf> p_attnVec;
    for(int32_t i =0; i<multiAttnData->nHeads_; i++)
    {
        p_attnVec.push_back(nn_softmax(scoreMatVec[i],0));
    }
    
    std::vector<MatrixXf> outputVec;
    for(int32_t i =0; i<multiAttnData->nHeads_; i++)
    {
        MatrixXf valueBlock = value.block(0,i*(int)(value.cols()/multiAttnData->nHeads_),
                                          value.rows(),(int)(value.cols()/multiAttnData->nHeads_));
                                          
        outputVec.push_back(p_attnVec[i]*valueBlock);
    }

    if(multiAttnData->winSize_ > 0)
    {
        std::vector<MatrixXf> relativeWeights = absolute_position_to_relative_position(p_attnVec);
        MatrixXf valueRelativeEmbeddings = get_relative_embeddings(multiAttnData->embRelV_,t_s);
        
        std::vector<MatrixXf> outMatTmpVec;

        for(int32_t i =0; i<multiAttnData->nHeads_; i++)
        {
            outputVec[i] = outputVec[i] + relativeWeights[i]*valueRelativeEmbeddings;
        }
    }
    
    MatrixXf ret = MatrixXf::Zero(outputVec[0].cols()*2,outputVec[0].rows());
    ret.block(0,0,outputVec[0].cols(),outputVec[0].rows()) = outputVec[0].transpose();
    ret.block(outputVec[1].cols(),0,outputVec[1].cols(),outputVec[1].rows()) = outputVec[1].transpose();

    MatrixXf retFlat = ret.reshaped(1,ret.rows()*ret.cols());
    MatrixXf retFlatReshaped = retFlat.reshaped(d,t_s);

    return retFlatReshaped;
}

multi_head_attention::~multi_head_attention()
{
    MODULE_MULTI_HEAD_ATTN_DATA_t * multiAttnData = (MODULE_MULTI_HEAD_ATTN_DATA_t* )priv_;

    delete multiAttnData->o_;
    delete multiAttnData->v_;
    delete multiAttnData->k_;
    delete multiAttnData->q_;
    delete multiAttnData;
}
