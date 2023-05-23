#include "iStft.h"
#include <unsupported/Eigen/FFT>
#include "stdio.h"
#include <complex> 
#include <iostream>

#define eps (1e-14)

using Eigen::Map;
using Eigen::MatrixXcf;

extern float hann_w[];
extern float hann_w_pow[];

typedef struct
{
    int32_t filterLen_;
    int32_t hopLen_;
    int32_t winLen_;

}ISFT_DATA_t;

iStft::iStft(int32_t filterLen, int32_t hopLen, int32_t winLen)
{
    ISFT_DATA_t * istftData = new ISFT_DATA_t();
    if(NULL == istftData)
    {
        return;
    }
    
    memset(istftData,0,sizeof(ISFT_DATA_t));

    istftData->filterLen_ = filterLen;
    istftData->hopLen_ = hopLen;
    istftData->winLen_ = winLen;
    
    priv_ = (void *) istftData;
}

iStft::~iStft()
{
    ISFT_DATA_t * istftData = (ISFT_DATA_t *)priv_;
    delete istftData;
}

MatrixXf iStft::forward(const MatrixXf & mag, const MatrixXf & phase)
{
    ISFT_DATA_t * istftData = (ISFT_DATA_t *)priv_;

    int32_t frames = mag.rows();
    int32_t oneSizeLen = mag.cols();

    MatrixXf ret = MatrixXf::Zero(1,(frames-1)*istftData->hopLen_+istftData->filterLen_);
    MatrixXf wsum = MatrixXf::Zero(1,(frames-1)*istftData->hopLen_+istftData->filterLen_);

    Eigen::FFT<float> fft;
    
    MatrixXf hann_mat = Map<MatrixXf>(hann_w,1,istftData->filterLen_);
    MatrixXf hann_pow_mat = Map<MatrixXf>(hann_w_pow,1,istftData->filterLen_);

    for(int32_t frameIdx = 0; frameIdx< (frames); frameIdx ++)
    {
        MatrixXcf frameC = MatrixXcf::Zero(1,istftData->filterLen_) ;

        for(int32_t i = 0; i< oneSizeLen; i++)
        {
            frameC.imag()(0,i) = phase(frameIdx,i);
        }

        frameC = frameC.array().exp();

        for(int32_t i = 0; i< oneSizeLen; i++)
        {
            frameC(0,i) = frameC(0,i)*mag(frameIdx,i);
        }

        int32_t s = 1;
        for(int32_t i = oneSizeLen; i < istftData->filterLen_; i++)
        {
            int32_t symIdx = i-(s++)*2;
            frameC.real()(0,i) = frameC.real()(0,symIdx);
            frameC.imag()(0,i) = frameC.imag()(0,symIdx)*(-1.0);
        }
        
        std::vector<std::complex<float> > freqvec;
        
        for(int32_t i = 0; i<istftData->filterLen_; i++)
        {
            std::complex<float> f(frameC.real()(0,i),frameC.imag()(0,i));
            freqvec.push_back(f);
        }

        std::vector<float> timevec;
        fft.inv(timevec,freqvec);

        MatrixXf frameMat(1,istftData->filterLen_);

        for(int32_t i = 0; i<istftData->filterLen_; i++)
        {
            frameMat(0,i) = timevec[i];            
        }

        frameMat = frameMat.array() * hann_mat.array();
        
        wsum.block(0,frameIdx*istftData->hopLen_,1,istftData->filterLen_) = 
        wsum.block(0,frameIdx*istftData->hopLen_,1,istftData->filterLen_).array() + hann_pow_mat.array();

        ret.block(0,frameIdx*istftData->hopLen_,1,istftData->filterLen_) = 
        ret.block(0,frameIdx*istftData->hopLen_,1,istftData->filterLen_) + frameMat;

    }

    for(int32_t i = 0; i<wsum.cols(); i++)
    {
        if(wsum(0,i) > eps)
        {
            ret(0,i) = ret(0,i)/wsum(0,i);
        }
    }
    
    MatrixXf retCentered = ret.block(0,(int32_t)(istftData->filterLen_/2),1,(frames-1)*istftData->hopLen_);

    return retCentered;
}


