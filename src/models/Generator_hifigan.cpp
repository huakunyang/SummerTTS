#include "Generator_hifigan.h"
#include "tts_logger.h"
#include "nn_conv1d.h"
#include "nn_conv1d_transposed.h"
#include "ResBlock1.h"
#include "nn_leaky_relu.h"
#include "nn_tanh.h"

using Eigen::Map;

typedef struct
{
    int32_t isMS_;
    int32_t upSampleRatesNum_;
    int32_t * upSampleRatesList_;
    int32_t upsample_initial_channel_;
    int32_t upSampleKernelSizesNum_;
    int32_t * upSampleKernelSizesList_;
    int32_t resBlocKernelSizeNum_;
    int32_t * resBlockKernelSizeList_;
    int32_t resBlockDilationSizeNum_;
    int32_t * resBlockDilationSizeList_;

    nn_conv1d * conv_pre_;
    nn_conv1d_transposed **upList_;
    ResBlock1 ** resBlockList_;
    nn_conv1d * conv_post_;
    nn_conv1d * cond_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}GENERATOR_DATA_t;

Generator_hifiGan::Generator_hifiGan(float * modelData, int32_t & offset,int32_t isMS)
{
    GENERATOR_DATA_t * generatorData = new GENERATOR_DATA_t();
    if(NULL == generatorData)
    {
        tts_log(TTS_LOG_ERROR, "Generator_hifigan: Failed to allocate memory for internal data block\n");
        return;
    }

    int32_t curOffset = offset;
    
    generatorData->isMS_ = isMS;
    generatorData->upSampleRatesNum_=(int32_t)modelData[curOffset++];
    generatorData->upSampleRatesList_ = new int32_t[generatorData->upSampleRatesNum_];
    for(int32_t i = 0; i<generatorData->upSampleRatesNum_; i++)
    {
        generatorData->upSampleRatesList_[i] = (int32_t)modelData[curOffset++];
    }
    generatorData->upsample_initial_channel_ = (int32_t)modelData[curOffset++];
    generatorData->upSampleKernelSizesNum_ = (int32_t)modelData[curOffset++]; 
    generatorData->upSampleKernelSizesList_ = new int32_t[generatorData->upSampleKernelSizesNum_];
    for(int32_t i = 0; i<generatorData->upSampleKernelSizesNum_; i++)
    {
        generatorData->upSampleKernelSizesList_[i] = modelData[curOffset++];
    }
    generatorData->resBlocKernelSizeNum_ = (int32_t)modelData[curOffset++];
    generatorData->resBlockKernelSizeList_ = new int32_t(generatorData->resBlocKernelSizeNum_);
    for(int32_t i = 0; i<generatorData->resBlocKernelSizeNum_; i++)
    {
        generatorData->resBlockKernelSizeList_[i] = (int32_t)modelData[curOffset++];
    }

    generatorData->resBlockDilationSizeNum_ = (int32_t)modelData[curOffset++]; 
    generatorData->resBlockDilationSizeList_ = new int32_t[generatorData->resBlockDilationSizeNum_*3];
    for(int32_t i = 0; i<generatorData->resBlockDilationSizeNum_; i++)
    {
        generatorData->resBlockDilationSizeList_[3*i+0] = (int32_t)modelData[curOffset++];
        generatorData->resBlockDilationSizeList_[3*i+1] = (int32_t)modelData[curOffset++];
        generatorData->resBlockDilationSizeList_[3*i+2] = (int32_t)modelData[curOffset++];
    }

    generatorData->conv_pre_ = new nn_conv1d(modelData, curOffset);
    generatorData->upList_ = (nn_conv1d_transposed**)malloc(sizeof(nn_conv1d_transposed *)*generatorData->upSampleRatesNum_);

    for(int32_t i = 0; i<generatorData->upSampleRatesNum_; i++)
    {
        int32_t u = generatorData->upSampleRatesList_[i];
        int32_t k = generatorData->upSampleKernelSizesList_[i];

        int32_t padding = floor((float)(k-u)/(2.0));
        generatorData->upList_[i] = new nn_conv1d_transposed(modelData, curOffset,u,padding); 
    }
    
    generatorData->resBlockList_ = (ResBlock1**)malloc(sizeof(ResBlock1 *)*(generatorData->upSampleRatesNum_ *
                                                                          generatorData->resBlocKernelSizeNum_));
   
    for(int32_t i = 0; i<generatorData->upSampleRatesNum_; i++)
    {
        for(int32_t j = 0; j<generatorData->resBlocKernelSizeNum_; j++)
        {
            generatorData->resBlockList_[i*generatorData->resBlocKernelSizeNum_+j] = new ResBlock1(modelData,curOffset);
        }
    }

    generatorData->conv_post_ = new nn_conv1d(modelData, curOffset); 
    generatorData->cond_ = NULL;
    if(generatorData->isMS_== 1)
    {
        generatorData->cond_ = new nn_conv1d(modelData, curOffset);
    }

    offset = curOffset;
    priv_ = (void *)generatorData;

}

Generator_hifiGan::~Generator_hifiGan()
{
    GENERATOR_DATA_t * generatorData = (GENERATOR_DATA_t *)priv_;
    delete [] generatorData->upSampleRatesList_;
    delete [] generatorData->upSampleKernelSizesList_;
    delete [] generatorData->resBlockKernelSizeList_;
    delete [] generatorData->resBlockDilationSizeList_;

    delete generatorData->conv_pre_;
    
    for(int32_t i = 0; i<generatorData->upSampleRatesNum_; i++)
    {
        delete generatorData->upList_[i];
    }
    free(generatorData->upList_);

    for(int32_t i = 0; i<generatorData->upSampleRatesNum_*generatorData->resBlocKernelSizeNum_; i++)
    {
        delete generatorData->resBlockList_[i];
    }
    free(generatorData->resBlockList_);

    if(generatorData->isMS_== 1)
    {
        delete generatorData->cond_;
    }
    delete generatorData->conv_post_;
    
    delete generatorData;
}

MatrixXf Generator_hifiGan::forward(const MatrixXf & x, const MatrixXf & g)
{
    GENERATOR_DATA_t * generatorData = (GENERATOR_DATA_t *)priv_;

    MatrixXf xx = generatorData->conv_pre_->forward(x);

    MatrixXf gg;
    if(generatorData->isMS_== 1)
    {
        gg = generatorData->cond_->forward(g);
        xx = xx.rowwise() + gg.row(0);
    }

    for(int32_t i = 0; i<generatorData->upSampleRatesNum_; i++)
    {
        xx = nn_leaky_relu(xx,0.1);
        xx = generatorData->upList_[i]->forward(xx);
        
        int32_t xs_none = 1;
        MatrixXf xs;
        for(int32_t j = 0; j<generatorData->resBlocKernelSizeNum_; j++)
        {
            if(xs_none == 1)
            {
                xs = generatorData->resBlockList_[i*generatorData->resBlocKernelSizeNum_ + j]->forward(xx);
                xs_none = 0;
            }
            else
            {
                MatrixXf xs_tmp = generatorData->resBlockList_[i*generatorData->resBlocKernelSizeNum_ + j]->forward(xx);
                xs = xs + xs_tmp;
            }

        }
        xx = xs.array()/(float)(generatorData->resBlocKernelSizeNum_);

    }

    xx = nn_leaky_relu(xx);
    xx = generatorData->conv_post_->forward(xx);
    xx = nn_tanh(xx);

    return xx;
}
