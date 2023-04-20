#include "nn_conv1d.h"
#include "tts_logger.h"

using Eigen::Map;

typedef struct
{
    int32_t outCh_;
    int32_t inCh_;
    int32_t kSize_;
    int32_t padding_;
    int32_t dilation_;
    int32_t hasBias_;
    int32_t sep_;

    MatrixXf w_;
    MatrixXf b_;
    
    int32_t print_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}NN_CONV1D_DATA_t;


int32_t parse_conv1d_parameter(float * modelData, int32_t & offset,
                               int32_t & inCh, int32_t & outCh, int32_t & kSize,
                               int32_t & padding, int32_t & dilation, int32_t & hasBias,
                               MatrixXf & weight, MatrixXf & bias)
{
    int32_t curOffset = offset;

    outCh = (int32_t)modelData[curOffset++];
    inCh = (int32_t)modelData[curOffset++];
    kSize = (int32_t)modelData[curOffset++];
    padding = (int32_t)modelData[curOffset++];
    dilation = (int32_t)modelData[curOffset++];
    hasBias = (int32_t)modelData[curOffset++];

    weight = Map<MatrixXf>(modelData+curOffset, inCh * kSize,outCh);
    curOffset = curOffset + inCh * kSize * outCh;

    if(1 == hasBias)
    {
        bias = Map<MatrixXf>(modelData+curOffset,1, outCh);
        curOffset = curOffset + 1*outCh;
    }

    offset = curOffset;

    return 1;
    
}

nn_conv1d::nn_conv1d(float * modelData, int32_t & offset, int32_t padding, int32_t dilation, int sep)
{
    NN_CONV1D_DATA_t * nn1dConvData = new NN_CONV1D_DATA_t();
    if(NULL == nn1dConvData)
    {
        tts_log(TTS_LOG_ERROR,"NN CONV1D: failed to allocate memory for internal block\n");
    }
    
    int32_t curOffset = offset;
    parse_conv1d_parameter(modelData, curOffset, nn1dConvData->inCh_, nn1dConvData->outCh_,
                                      nn1dConvData->kSize_, nn1dConvData->padding_,
                                      nn1dConvData->dilation_, nn1dConvData->hasBias_,
                                      nn1dConvData->w_, nn1dConvData->b_);

    nn1dConvData->padding_ = padding;
    nn1dConvData->dilation_ = dilation;
    nn1dConvData->sep_ = sep;

    offset = curOffset;
    priv_ = (void*)nn1dConvData;
}

nn_conv1d::nn_conv1d(float * modelData, int32_t & offset)
{
    NN_CONV1D_DATA_t * nn1dConvData = new NN_CONV1D_DATA_t();
    if(NULL == nn1dConvData)
    {
        tts_log(TTS_LOG_ERROR,"NN CONV1D: failed to allocate memory for internal block\n");
    }
    
    int32_t curOffset = offset;
    parse_conv1d_parameter(modelData, curOffset, nn1dConvData->inCh_, nn1dConvData->outCh_,
                                      nn1dConvData->kSize_, nn1dConvData->padding_,
                                      nn1dConvData->dilation_, nn1dConvData->hasBias_,
                                      nn1dConvData->w_, nn1dConvData->b_);
    nn1dConvData->sep_ = 0;

    offset = curOffset;
    priv_ = (void*)nn1dConvData;
}

nn_conv1d::nn_conv1d(int32_t inCh, int32_t outCh, int32_t kSize, 
          int32_t padding, int32_t dilation,int hasBias, MatrixXf & weight, MatrixXf & bias)
{
    NN_CONV1D_DATA_t * nn1dConvData = new NN_CONV1D_DATA_t();
    if(NULL == nn1dConvData)
    {
        tts_log(TTS_LOG_ERROR,"NN CONV1D: failed to allocate memory for internal block\n");
        return;
    }

    nn1dConvData->inCh_ = inCh;
    nn1dConvData->outCh_ = outCh;
    nn1dConvData->kSize_ = kSize;
    nn1dConvData->padding_ = padding;
    nn1dConvData->dilation_ = dilation;
    nn1dConvData->hasBias_ = hasBias;
    nn1dConvData->w_ = weight;
    nn1dConvData->b_ = bias;

    priv_ = (void*)nn1dConvData;

}

MatrixXf nn_conv1d::forward(MatrixXf inputMat)
{
    NN_CONV1D_DATA_t * nn1dConvData = (NN_CONV1D_DATA_t *)priv_;

    MatrixXf paddingMat = MatrixXf::Zero(inputMat.rows()+2*nn1dConvData->padding_,inputMat.cols());
    
    if(nn1dConvData->padding_ == 0)
    {
        paddingMat = inputMat;
    }
    else
    {
        paddingMat.block(nn1dConvData->padding_,0,inputMat.rows(),inputMat.cols()) = inputMat;
    }

    int32_t dilatedKSize = nn1dConvData->w_.rows();
    if(nn1dConvData->dilation_ !=1)
    {
        dilatedKSize = nn1dConvData->inCh_*((nn1dConvData->kSize_-1)*(nn1dConvData->dilation_-1)+nn1dConvData->kSize_);
    }

    MatrixXf dilationMat = MatrixXf::Zero(dilatedKSize,nn1dConvData->w_.cols());

    if(nn1dConvData->dilation_ == 1)
    {
        dilationMat = nn1dConvData->w_;
    }
    else
    {
        MatrixXf newKernel = MatrixXf::Zero(dilatedKSize,nn1dConvData->w_.cols());
        for(int32_t i = 0; i<nn1dConvData->kSize_; i++)
        {
            newKernel.block(i*(nn1dConvData->dilation_)*nn1dConvData->inCh_,0
                            ,nn1dConvData->inCh_,nn1dConvData->w_.cols()) = 
            nn1dConvData->w_.block(i*nn1dConvData->inCh_,0,nn1dConvData->inCh_,nn1dConvData->w_.cols());
        }
        dilationMat = newKernel;
    }
    
    int32_t outLen =  paddingMat.rows() - nn1dConvData->dilation_*(nn1dConvData->kSize_ -1);

    MatrixXf result = MatrixXf::Zero(outLen,nn1dConvData->outCh_);
    
    int32_t newKSize = nn1dConvData->kSize_;
    if(nn1dConvData->dilation_ != 1)
    {
        newKSize = (nn1dConvData->kSize_-1)*(nn1dConvData->dilation_-1)+nn1dConvData->kSize_;
    }
    
    if(nn1dConvData->sep_ != 0)
    {
        for(int32_t i = 0; i<outLen; i++)
        {
        
            MatrixXf InputBlock = paddingMat.block(i,0,newKSize,nn1dConvData->outCh_).transpose();
            for(int32_t j = 0; j < nn1dConvData->outCh_; j++)
            {
                result(i,j) = InputBlock.row(j)*dilationMat.col(j);
            }
        }

    }
    else
    {
        MatrixXf inputMatForCalc = MatrixXf::Zero(outLen,newKSize*nn1dConvData->inCh_);

        for(int32_t i = 0; i<outLen; i++)
        {
            inputMatForCalc.row(i) = paddingMat.block(i,0,newKSize,nn1dConvData->inCh_).reshaped<Eigen::RowMajor>().transpose();
        }

        result = inputMatForCalc * dilationMat;
        
    }

    if(1 == nn1dConvData->hasBias_)
    {
        result = result.rowwise() + nn1dConvData->b_.row(0);
    }
    
    return result;
}

void nn_conv1d::print_p()
{
    NN_CONV1D_DATA_t * nn1dConvData = (NN_CONV1D_DATA_t *)priv_;
    nn1dConvData->print_ = 1;
}

int32_t nn_conv1d::get_in_channels_num()
{
    NN_CONV1D_DATA_t * nn1dConvData = (NN_CONV1D_DATA_t *)priv_;
    return nn1dConvData->inCh_; 
}

int32_t nn_conv1d::get_out_channels_num()
{
    NN_CONV1D_DATA_t * nn1dConvData = (NN_CONV1D_DATA_t *)priv_;
    return nn1dConvData->outCh_; 
}

nn_conv1d::~nn_conv1d()
{

    NN_CONV1D_DATA_t * nn1dConvData = (NN_CONV1D_DATA_t *)priv_;
    delete nn1dConvData;
}
