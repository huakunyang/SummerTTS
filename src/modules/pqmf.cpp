#include "pqmf.h"
#include "nn_conv1d.h"
#include "nn_conv1d_transposed.h"
#include "tts_logger.h"

using Eigen::Map;

float h_filter[] = {
 8.36595339e-06, 2.68017852e-05,  5.05711124e-05,  6.13482515e-05,
 2.75281598e-05, -8.62839965e-05, -2.99268467e-04, -5.88389492e-04,
 -8.67064627e-04, -9.82905838e-04, -7.47200209e-04,  8.04087656e-19,
 1.30001234e-03,  2.98798828e-03,  4.64603942e-03,  5.63488600e-03,
 5.22586317e-03,  2.82493436e-03, -1.75650987e-03, -8.06073440e-03,
 -1.48622207e-02, -2.02404650e-02, -2.18780344e-02, -1.75512321e-02,
 -5.71474631e-03,  1.39652689e-02,  4.02848855e-02,  7.05021626e-02,
 1.00706377e-01,  1.26503321e-01,  1.43873012e-01,  1.50000000e-01,
 1.43873012e-01,  1.26503321e-01,  1.00706377e-01,  7.05021626e-02,
 4.02848855e-02,  1.39652689e-02, -5.71474631e-03, -1.75512321e-02,
 -2.18780344e-02, -2.02404650e-02, -1.48622207e-02,  -8.06073440e-03,
 -1.75650987e-03,  2.82493436e-03,  5.22586317e-03,  5.63488600e-03,
 4.64603942e-03,  2.98798828e-03,  1.30001234e-03,  8.04087656e-19,
 -7.47200209e-04, -9.82905838e-04, -8.67064627e-04, -5.88389492e-04,
 -2.99268467e-04, -8.62839965e-05,  2.75281598e-05,  6.13482515e-05,
 5.05711124e-05,  2.68017852e-05,  8.36595339e-06
 };

typedef struct
{
    int32_t subBands_;
    MatrixXf h_synthesis_;
    MatrixXf updown_filter_;

    nn_conv1d_transposed * upDownConv_;
    nn_conv1d * conv1d_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}PQMF_DATA_t;

pqmf::pqmf(int32_t subBands)
{
    PQMF_DATA_t * pqmfData = new PQMF_DATA_t();

    if(NULL == pqmfData)
    {
        tts_log(TTS_LOG_ERROR, "PQMF: Failed to allocate memory for internal data block\n");
        return;
    }

    pqmfData->subBands_ = subBands;
    
    int32_t filterLen = sizeof(h_filter)/sizeof(float);

    MatrixXf h_proto = Map<MatrixXf>(h_filter, 1,filterLen);

    int32_t taps = 62;

    MatrixXf tapsMat = MatrixXf::Zero(taps+1,1);
    for(int32_t itap = 0; itap <(taps+1); itap++)
    {
        tapsMat(itap,0) = itap;     
    }

    MatrixXf tmp1 = (tapsMat.array()-(((float)taps-1)/2))*(M_PI/(2*subBands));

    MatrixXf ng1 = MatrixXf::Zero(1,1);
    ng1(0,0) = -1;


    MatrixXf h_syntmp = MatrixXf::Zero(subBands,filterLen);
    for(int32_t idx = 0; idx < subBands; idx++)
    {
        MatrixXf tmp2 = (tmp1.array()*(2*idx+1) - ((ng1.array().pow(idx)*(M_PI/4))(0,0))).array().cos();
        tmp2 = tmp2.transpose();
        h_syntmp.row(idx) = h_proto.array()*2*tmp2.array();
    }

    pqmfData->h_synthesis_ = h_syntmp.reshaped(filterLen*subBands,1);

    pqmfData->updown_filter_ = MatrixXf::Zero(subBands,subBands*subBands);

    for(int32_t i = 0; i< subBands; i++)
    {
        pqmfData->updown_filter_(i,i*4) = subBands;    
    }

    MatrixXf biasDummy;
    pqmfData->upDownConv_ = new nn_conv1d_transposed(subBands,
                                                          subBands,
                                                          subBands,
                                                          0,
                                                          1,
                                                          0,
                                                          subBands,
                                                          pqmfData->updown_filter_,
                                                          biasDummy);

    MatrixXf biasDummy2;
    MatrixXf h_synthesis_t = pqmfData->h_synthesis_;
    pqmfData->conv1d_ = new nn_conv1d(subBands,1,filterLen,0,1,0,h_synthesis_t ,biasDummy2);
    priv_ = (void *)pqmfData;

}

MatrixXf pqmf::forward(const MatrixXf & inputMat)
{
    PQMF_DATA_t * pqmfData = (PQMF_DATA_t*)priv_;

    MatrixXf xx = pqmfData->upDownConv_->forward(inputMat);

    MatrixXf xx_pad = MatrixXf::Zero(xx.rows()+2*31,xx.cols());
    xx_pad.block(31,0,xx.rows(),xx.cols()) = xx;

    xx = pqmfData->conv1d_->forward(xx_pad);
    return xx;
}

pqmf::~pqmf()
{
    PQMF_DATA_t * pqmfData = (PQMF_DATA_t*)priv_;
    delete pqmfData;
}

