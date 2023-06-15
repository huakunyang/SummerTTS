#include "SynthesizerTrn.h"
#include "TextEncoder.h"
#include "StochasticDurationPredictor.h"
#include "FixDurationPredictor.h"
#include "ResidualCouplingBlock.h"
#include "Generator_hifigan.h"
#include "Generator_MS.h"
#include "Generator_Istft.h"
#include "Generator_MBB.h"
#include "DurationPredictor_base.h"
#include "FixDurationPredictor.h"
#include "StochasticDurationPredictor.h"
#include "tts_logger.h"
#include "nn_clamp_min.h"
#include "random_gen.h"
#include <Eigen/Dense>
#include "hanzi2phoneid.h"
#include "processor/processor.h"
#include "utils/flags.h"
#include <iostream>
#include <streambuf>
#include "cppjieba/Jieba.hpp"
#include "EnglishText2Id.h"

using Eigen::MatrixXf;
using Eigen::Map;

struct membuf : std::streambuf
{
    membuf(char* begin, char* end) 
    {
        this->setg(begin, begin, end);
    }
};

typedef struct
{
    int32_t isMS_;
    int32_t langType_;
    int32_t durPredType_;
    int32_t decType_;
    int32_t spkNum_;
    int32_t gin_channels_;
    vector<string> jieba_words_;
    
    hanzi2phoneid * hz2ID_;
    TextEncoder * textEncoder_;
    DurationPredictor_base * durPredicator_;
    ResidualCouplingBlock * flow_;
    Generator_base * dec_;
    MatrixXf emg_;
    wetext::Processor * tnProcessor_;
    cppjieba::Jieba * jieba_;
    
    EnglishText2Id * eng2Ipa_;
}SYN_DATA_t;

typedef enum
{
    LANG_TYPE_CHS=0,
    LANG_TYPE_ENG=1

}LANG_TYPE_t;

typedef enum
{
    DUR_PRED_TYPE_STOC = 0,
    DUR_PRED_TYPE_NOSTOC = 1
}DUR_PRED_TYPE_t;

typedef enum
{
    DEC_TYPE_HIFIGAN = 0,
    DEC_TYPE_MS = 1,
    DEC_TYPE_ISTFT = 2,
    DEC_TYPE_MBB = 3
}DEC_TYPE_t;

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

SynthesizerTrn::SynthesizerTrn(float * modelData, int32_t modelSize)
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
    synData->langType_ = (int32_t)modelData[offset++]; 
    synData->durPredType_ = (int32_t)modelData[offset++]; 
    synData->decType_ = (int32_t)modelData[offset++];

    synData->textEncoder_ = new TextEncoder(modelData,offset);

    if(synData->decType_ == DEC_TYPE_HIFIGAN)
    {
        synData->dec_ = new Generator_hifiGan(modelData, offset,synData->isMS_);
    }
    else if(synData->decType_ == DEC_TYPE_MS)
    {
        synData->dec_ = new Generator_MS(modelData, offset,synData->isMS_);
    }
    else if(synData->decType_ == DEC_TYPE_ISTFT)
    {
        synData->dec_ = new Generator_Istft(modelData, offset,synData->isMS_);
    }
    else if(synData->decType_ == DEC_TYPE_MBB)
    {
        synData->dec_ = new Generator_MBB(modelData, offset,synData->isMS_);
    }
    else
    {
        tts_log(TTS_LOG_ERROR, "SynthesizerTrn: Unknown decoder \n");
        delete synData->textEncoder_;
        delete synData;
        return;
    }
    
    synData->flow_ = new ResidualCouplingBlock(modelData,offset,1,synData->isMS_);

    if(synData->durPredType_ == DUR_PRED_TYPE_STOC)
    {
        synData->durPredicator_ = new StochasticDurationPredictor(modelData, offset,synData->isMS_);
    }
    else if(synData->durPredType_ == DUR_PRED_TYPE_NOSTOC)
    {
        synData->durPredicator_ = new FixDurationPredictor(modelData, offset,synData->isMS_);;
    }
    else
    {
        tts_log(TTS_LOG_ERROR, "SynthesizerTrn: Unknown duration predicator \n");
        delete synData->dec_;
        delete synData->textEncoder_;
        delete synData->flow_;
        delete synData;
        return;
    }


    if(synData->isMS_ == 1)
    {
        synData->spkNum_ = (int32_t)modelData[offset++];
        synData->gin_channels_ = (int32_t)modelData[offset++];
        synData->emg_ = Map<MatrixXf>(modelData+offset,synData->spkNum_, synData->gin_channels_);

        synData->durPredicator_->setMSSpk(synData->isMS_, synData->gin_channels_);
        offset = offset+synData->spkNum_*synData->gin_channels_;
    }
    else
    {
        synData->durPredicator_->setMSSpk(0,0);
    }
    
    if(synData->langType_ == LANG_TYPE_ENG)
    {
        if(modelSize > (offset +1)*sizeof(float))
        {
            int32_t curOffset = 0;
            synData->eng2Ipa_ = new EnglishText2Id(modelData+offset, curOffset);
        }
    }
    else if(synData->langType_ == LANG_TYPE_CHS)
    {
        int32_t offset_char = offset*sizeof(float);
    
        if(offset*sizeof(float)+1 < modelSize)
        {
            int32_t tnTaggerSize = (int32_t)modelData[offset++];
            int32_t tnVerSize = (int32_t)modelData[offset++];

            membuf sbufTagger((char *)(modelData + offset), (char *)(modelData + offset) + tnTaggerSize);

            membuf sbufVerb((char *)(modelData + offset)+tnTaggerSize, (char *)(modelData + offset) + tnTaggerSize + tnVerSize);

            offset_char = offset*sizeof(float) + tnTaggerSize + tnVerSize;

            if(offset_char % (sizeof(float)) > 0)
            {
                offset_char = offset_char + (offset_char % (sizeof(float)));
            }

            offset = offset_char/sizeof(float);

            std::istream inTagger(&sbufTagger);
            std::istream inVerb(&sbufVerb);

            synData->tnProcessor_ = new wetext::Processor(inTagger, inVerb);
        }
        else
        {
            synData->tnProcessor_ = NULL;
        }

        if((offset_char +1) < modelSize)
        {
            int32_t jiebaDictSize = (int32_t)modelData[offset++];
            int32_t jiebaHmmModelSize = (int32_t)modelData[offset++];
            int32_t jiebaUsrdictSize = (int32_t)modelData[offset++];
            int32_t jiebaIdfSize = (int32_t)modelData[offset++];
            int32_t jiebaStopwordSize = (int32_t)modelData[offset++];
        
            membuf sbufDict((char *)(modelData + offset), (char *)(modelData + offset) + 
                                                              jiebaDictSize);

            membuf sbufModel((char *)(modelData + offset)+
                                 jiebaDictSize, (char *)(modelData + offset) + 
                                                jiebaDictSize + 
                                                jiebaHmmModelSize);

            membuf sbufUsrDict((char *)(modelData + offset)+
                                   jiebaDictSize+
                                   jiebaHmmModelSize, (char *)(modelData + offset) + 
                                                                       jiebaDictSize + 
                                                                       jiebaHmmModelSize +
                                                                       jiebaUsrdictSize );
                                                                       
            membuf sbufIdf((char *)(modelData + offset)+
                                jiebaDictSize +
                                jiebaHmmModelSize+
                                jiebaUsrdictSize,    (char *)(modelData + offset) + 
                                                             jiebaDictSize + 
                                                             jiebaHmmModelSize +
                                                             jiebaUsrdictSize +
                                                             jiebaIdfSize);
                                                                           
            membuf sbufStopword((char *)(modelData + offset)+
                            jiebaDictSize+
                            jiebaHmmModelSize+
                            jiebaUsrdictSize+
                            jiebaIdfSize ,    (char *)(modelData + offset) + 
                                                jiebaDictSize + 
                                                jiebaHmmModelSize +
                                                jiebaUsrdictSize +
                                                jiebaIdfSize +
                                                jiebaStopwordSize);
            std::istream inDict(&sbufDict);
            std::istream inModel(&sbufModel);
            std::istream inUsrDict(&sbufUsrDict);
            std::istream inIdf(&sbufIdf);
            std::istream inStopword(&sbufStopword);



            synData->jieba_ = new cppjieba::Jieba(inDict, inModel,
                                              inUsrDict, inIdf, inStopword);

            offset_char = offset*sizeof(float) + jiebaDictSize + jiebaHmmModelSize + jiebaUsrdictSize + jiebaIdfSize + jiebaStopwordSize;

            if(offset_char % (sizeof(float)) > 0)
            {
                offset_char = offset_char + (offset_char % (sizeof(float)));
            }

            offset = offset_char/sizeof(float);
        }

        if((offset_char +1) < modelSize)
        {
            int32_t multiPhoneWordSize = (int32_t)modelData[offset++];
            int32_t multiPhonePinyinSize = (int32_t)modelData[offset++];
        
            membuf sbufMultiPhoneWords((char *)(modelData + offset), (char *)(modelData + offset) + 
                                                                  multiPhoneWordSize);

            membuf sbufMultiPhoneWordsPinyin((char *)(modelData + offset)+
                                                    multiPhoneWordSize, (char *)(modelData + offset) + 
                                                                        multiPhoneWordSize + 
                                                                        multiPhonePinyinSize);

            std::istream inMultiPhonewords(&sbufMultiPhoneWords);
            std::istream inMultiPhonewordsPinyin(&sbufMultiPhoneWordsPinyin);
            synData->hz2ID_ = new hanzi2phoneid(inMultiPhonewords,inMultiPhonewordsPinyin);

            offset_char = offset*sizeof(float) + multiPhoneWordSize + multiPhonePinyinSize;

            if(offset_char % (sizeof(float)) > 0)
            {
                offset_char = offset_char + (offset_char % (sizeof(float)));
            }

            offset = offset_char/sizeof(float);
        }
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

    int32_t * strIDs = NULL;
    int32_t strLen = 0;
    if(synData->langType_ == LANG_TYPE_CHS)
    {
        string tnString = line;
        if(synData->tnProcessor_ != NULL)
        {
            string tagged_text = synData->tnProcessor_->tag(line);
            tnString = synData->tnProcessor_->verbalize(tagged_text);
        }

        synData->jieba_->Cut(tnString, synData->jieba_words_, true);

        strIDs = synData->hz2ID_->convert(tnString,strLen, synData->jieba_words_);

    }
    else if(synData->langType_ == LANG_TYPE_ENG)
    {
        vector<int> engIDVec = synData->eng2Ipa_->getIPAId(line);
        strLen = engIDVec.size();

        strIDs = new int32_t[strLen];

        for(int32_t ii = 0; ii<strLen; ii++)
        {
            strIDs[ii] = engIDVec[ii];
        }
        lengthScale = lengthScale*0.83;
    }

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

    MatrixXf logw = synData->durPredicator_->forward(XX,g,noiseScale);
    
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
    delete synData->durPredicator_;
    delete synData->flow_;
    delete synData->dec_;
    delete synData->hz2ID_;
    delete synData->tnProcessor_;
    delete synData->jieba_;
    delete synData->eng2Ipa_;
    delete synData;
}


