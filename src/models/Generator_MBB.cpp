#include "Generator_MBB.h"
#include "tts_logger.h"
#include "nn_conv1d.h"
#include "nn_conv1d_transposed.h"
#include "ResBlock1.h"
#include "nn_leaky_relu.h"
#include "nn_tanh.h"
#include "iStft.h"
#include "pqmf.h"
#include <vector>

using Eigen::Map;

typedef struct
{
    int32_t isMS_;
    int32_t subBands_;
    int32_t gen_istft_n_fft_;
    int32_t gen_istft_hop_size_;
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
    nn_conv1d * subband_conv_post_;
    iStft * istft_;
    pqmf * pqmf_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}GENERATOR_MBB_DATA_t;

Generator_MBB::Generator_MBB(float * modelData, int32_t & offset,int32_t isMS)
{
    GENERATOR_MBB_DATA_t * generatorData = new GENERATOR_MBB_DATA_t();
    if(NULL == generatorData)
    {
        tts_log(TTS_LOG_ERROR, "Generator: Failed to allocate memory for internal data block\n");
        return;
    }

    int32_t curOffset = offset;

    generatorData->isMS_ = isMS;
    generatorData->subBands_=(int32_t)modelData[curOffset++];
    generatorData->gen_istft_n_fft_ =(int32_t)modelData[curOffset++];
    generatorData->gen_istft_hop_size_=(int32_t)modelData[curOffset++];
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

    generatorData->subband_conv_post_ = new nn_conv1d(modelData, curOffset); 
    generatorData->istft_ = new iStft(16,4,16);
    generatorData->pqmf_ = new pqmf(4);

    offset = curOffset;
    priv_ = (void *)generatorData;
}

Generator_MBB::~Generator_MBB()
{
    GENERATOR_MBB_DATA_t * generatorData = (GENERATOR_MBB_DATA_t *)priv_;
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

    delete generatorData->subband_conv_post_;
    delete generatorData->istft_;
    delete generatorData->pqmf_;

    delete generatorData;
}

MatrixXf Generator_MBB::forward(const MatrixXf & x, const MatrixXf & g)
{
    GENERATOR_MBB_DATA_t * generatorData = (GENERATOR_MBB_DATA_t *)priv_;

    MatrixXf xx = x;

    xx = generatorData->conv_pre_->forward(xx);

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

    MatrixXf xx_refpad = MatrixXf::Zero(xx.rows()+1,xx.cols());
    xx_refpad.block(1,0,xx.rows(),xx.cols()) = xx;

    if(xx.rows() > 1)
    {
        xx_refpad.row(0) = xx.row(1);
    }

    xx = generatorData->subband_conv_post_->forward(xx_refpad);
    
    MatrixXf timeMat = MatrixXf::Zero(((xx.rows()-1)*4),generatorData->subBands_);

    int32_t subBandCols = xx.cols()/generatorData->subBands_;
    int32_t halfSubBandCols = generatorData->gen_istft_n_fft_/2+1;

    for(int32_t bIdx = 0; bIdx < generatorData->subBands_; bIdx++)
    {
        MatrixXf subBandX = xx.block(0,bIdx*subBandCols,xx.rows(),subBandCols);
        MatrixXf subBandp1 = subBandX.block(0,0,subBandX.rows(),halfSubBandCols);
        MatrixXf subBandp2 = subBandX.block(0,halfSubBandCols,subBandX.rows(),halfSubBandCols);
        MatrixXf subBandTime = generatorData->istft_->forward((subBandp1.array().exp()).matrix(),
                                                              ((subBandp2.array().sin())*M_PI).matrix());
        timeMat.col(bIdx) = subBandTime.transpose();
    }

    xx = generatorData->pqmf_->forward(timeMat);
    return xx;
}
