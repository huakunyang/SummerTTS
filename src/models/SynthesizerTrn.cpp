#include "SynthesizerTrn.h"
#include "TextEncoder.h"
#include "StochasticDurationPredictor.h"
#include "ResidualCouplingBlock.h"
#include "Generator.h"
#include "tts_logger.h"
#include "nn_clamp_min.h"
#include "random_gen.h"
#include <Eigen/Dense>
#include "hanzi2phoneid.h"

using Eigen::MatrixXf;
using Eigen::Map;

typedef struct
{
    int32_t isMS_;
    int32_t spkNum_;
    int32_t gin_channels_;
    hanzi2phoneid * hz2ID_;
    TextEncoder * textEncoder_;
    StochasticDurationPredictor * stochDurPred_;
    ResidualCouplingBlock * flow_;
    Generator * dec_;
    MatrixXf emg_;

}SYN_DATA_t;

int32_t SynthesizerTrn::getSpeakerNum()
{
    SYN_DATA_t * synData = (SYN_DATA_t *)priv_;
    int32_t spkNum = synData->spkNum_;
    if(spkNum == 0)
    {
        spkNum = 1;
    }
    return spkNum;
}

SynthesizerTrn::SynthesizerTrn(float * modelData)
{
    SYN_DATA_t * synData = new SYN_DATA_t();
    if(NULL == synData)
    {
        tts_log(TTS_LOG_ERROR, "SynthesizerTrn: Failed to allocate memory for internal data block\n");
        return;
    }
    memset(synData, 0, sizeof(SYN_DATA_t));

    int32_t offset = 0;
    synData->isMS_ = (int32_t)modelData[offset++]; 
    synData->textEncoder_ = new TextEncoder(modelData,offset);
    synData->dec_ = new Generator(modelData, offset,synData->isMS_);
    synData->flow_ = new ResidualCouplingBlock(modelData,offset,1,synData->isMS_);
    synData->stochDurPred_ = new StochasticDurationPredictor(modelData, offset,synData->isMS_);

    synData->hz2ID_ = new hanzi2phoneid();

    if(synData->isMS_ == 1)
    {
        synData->spkNum_ = (int32_t)modelData[offset++];
        synData->gin_channels_ = (int32_t)modelData[offset++];
        synData->emg_ = Map<MatrixXf>(modelData+offset,synData->spkNum_, synData->gin_channels_);
        synData->stochDurPred_->setMSSpk(synData->isMS_, synData->gin_channels_);
    }
    else
    {
        synData->stochDurPred_->setMSSpk(0, 0);
    }
                                     
    priv_ = synData;
}

MatrixXf expandM(const MatrixXf & x, const MatrixXf lengthM)
{
    MatrixXf y_lengths = nn_clamp_min(lengthM.colwise().sum(),1.0);
    int32_t totalLen = (int32_t)y_lengths(0,0);

    MatrixXf ret = MatrixXf::Zero(totalLen,x.cols());

    int32_t rowIdx = 0;
    for(int32_t i = 0; i<lengthM.rows(); i++)
    {
        int32_t len = (int32_t)lengthM(i,0);
        for(int32_t j = 0; j<len; j++)
        {
            ret.row(rowIdx++) = x.row(i);
        }
    }
    return ret; 
}

int16_t * SynthesizerTrn::infer(const string & line, int32_t sid, float lengthScale, int32_t & dataLen)
{
    SYN_DATA_t * synData = (SYN_DATA_t *)priv_;

    int32_t strLen;
    int32_t * strIDs = synData->hz2ID_->convert(line,strLen);

    float noiseScale = 0.0;

    MatrixXf m;
    MatrixXf logs;
    MatrixXf XX = synData->textEncoder_->forward(strIDs,strLen,m,logs);
    
    MatrixXf g;
    if(synData->isMS_ == 1)
    {
        if((sid<0) || (sid >= synData->spkNum_))
        {
            sid = 0;
        }

        g = synData->emg_.row(sid); 
    }

    MatrixXf logw = synData->stochDurPred_->forward(XX,g,noiseScale); 

    MatrixXf w = logw.array().exp() * lengthScale;
    MatrixXf w_ceil = w.array().ceil();
    MatrixXf y_lengths = nn_clamp_min(w_ceil.colwise().sum(),1.0);
    
    MatrixXf m_expand = expandM(m, w_ceil);
    MatrixXf logs_expand = expandM(logs,w_ceil);
    
    MatrixXf z_p = m_expand.array() + rand_gen(m_expand.rows(), m_expand.cols(), 0.0, 1.0).array() * logs_expand.array() * noiseScale;

    MatrixXf z = synData->flow_->forward(z_p,g);

    MatrixXf o = synData->dec_->forward(z,g); 

    dataLen = o.rows()*o.cols();

    int16_t * retData = (int16_t *)malloc(sizeof(int16_t)*dataLen);

    for(int32_t i = 0; i< dataLen; i++)
    {
        retData[i] = (int16_t)(o.data()[i]*32737);
    }

    delete [] strIDs;

    return retData;
}

SynthesizerTrn::~SynthesizerTrn()
{
    SYN_DATA_t * synData = (SYN_DATA_t *)priv_;
    delete synData->textEncoder_;
    delete synData->stochDurPred_;
    delete synData->flow_;
    delete synData->dec_;
    delete synData->hz2ID_;
    delete synData;
}


