#include "ResidualCouplingLayer.h"
#include "tts_logger.h"
#include "nn_conv1d.h"
#include "WN.h"

typedef struct
{
    int32_t isMS_;
    nn_conv1d * pre_; 
    WN * enc_;
    nn_conv1d * post_;

}RESIDUAL_COUPLING_LAYER_DATA_t;

ResidualCouplingLayer::ResidualCouplingLayer(float * modelData, int32_t & offset, int32_t dilation_rate, int32_t isMS)
{
    RESIDUAL_COUPLING_LAYER_DATA_t * residualCouplingLayer = new RESIDUAL_COUPLING_LAYER_DATA_t();
    if(NULL == residualCouplingLayer)
    {
        tts_log(TTS_LOG_ERROR, "ResidualCouplingLayer: Failed to allocate memory for internal data block\n");
        return;
    }
    
    memset(residualCouplingLayer,0,sizeof(RESIDUAL_COUPLING_LAYER_DATA_t));
    
    int32_t curOffset = offset;
    residualCouplingLayer->isMS_ = isMS;
    residualCouplingLayer->pre_ = new nn_conv1d(modelData,curOffset);
    residualCouplingLayer->enc_ = new WN(modelData,curOffset,dilation_rate,residualCouplingLayer->isMS_);
    residualCouplingLayer->post_ = new nn_conv1d(modelData,curOffset);

    offset = curOffset;
    priv_ = (void *)residualCouplingLayer;

}

ResidualCouplingLayer::~ResidualCouplingLayer()
{
    RESIDUAL_COUPLING_LAYER_DATA_t * residualCouplingLayer = (RESIDUAL_COUPLING_LAYER_DATA_t *)priv_;

    delete residualCouplingLayer->pre_;
    delete residualCouplingLayer->enc_;
    delete residualCouplingLayer->post_;
    delete residualCouplingLayer;
}

MatrixXf ResidualCouplingLayer::forward(const MatrixXf & x, const MatrixXf & g)
{
    RESIDUAL_COUPLING_LAYER_DATA_t * residualCouplingLayer = (RESIDUAL_COUPLING_LAYER_DATA_t *)priv_;
    
    MatrixXf x0 = x.block(0,0, x.rows(), (int32_t)(x.cols()/2));
    MatrixXf x1 = x.block(0,(int32_t)(x.cols()/2),x.rows(), (int32_t)(x.cols()/2));

    MatrixXf h = residualCouplingLayer->pre_->forward(x0);
    h = residualCouplingLayer->enc_->forward(h,g);

    MatrixXf m = residualCouplingLayer->post_->forward(h);

    x1 = x1 - m;
    MatrixXf ret = MatrixXf::Zero(x.rows(),x.cols());

    ret.block(0,0,x0.rows(),x0.cols()) = x0;
    ret.block(0,x0.cols(),x1.rows(),x1.cols()) = x1;

    return ret;
}
