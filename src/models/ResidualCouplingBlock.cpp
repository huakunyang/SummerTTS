#include "ResidualCouplingBlock.h"
#include "tts_logger.h"
#include "nn_flip.h"
#include "ResidualCouplingLayer.h"

typedef struct
{
    int32_t n_flows_;
    int32_t n_layers_;
    int32_t channels_;
    int32_t hidden_channels_;
    int32_t isMS_;
    ResidualCouplingLayer ** residualCouplingLayerList_;

}RESIDUAL_COUPLING_BLOCK_DATA_t;

ResidualCouplingBlock::ResidualCouplingBlock(float * modelData, int32_t & offset, int32_t dilation_rate,int32_t isMS)
{
    RESIDUAL_COUPLING_BLOCK_DATA_t * residualCouplingBlockData = new RESIDUAL_COUPLING_BLOCK_DATA_t();
    if(NULL == residualCouplingBlockData)
    {
        tts_log(TTS_LOG_ERROR, "ResidualCouplingBlock: Failed to allocate memory for internal data block\n");
        return;
    }
    memset(residualCouplingBlockData, 0, sizeof(RESIDUAL_COUPLING_BLOCK_DATA_t));

    residualCouplingBlockData->isMS_ = isMS;
    int32_t curOffset = offset;
    residualCouplingBlockData->n_flows_ = (int32_t)modelData[curOffset++];
    residualCouplingBlockData->n_layers_ = (int32_t)modelData[curOffset++];

    residualCouplingBlockData->residualCouplingLayerList_ = (ResidualCouplingLayer **)(malloc(sizeof(RESIDUAL_COUPLING_BLOCK_DATA_t *) *
                                                                   residualCouplingBlockData->n_flows_));
    
    
    for(int32_t i = 0; i<residualCouplingBlockData->n_flows_; i++)
    {
        residualCouplingBlockData->residualCouplingLayerList_[i] = new ResidualCouplingLayer(modelData, curOffset,dilation_rate, isMS);
    }

    offset = curOffset;
    priv_ = (void*)residualCouplingBlockData;
    
}

ResidualCouplingBlock::~ResidualCouplingBlock()
{
    RESIDUAL_COUPLING_BLOCK_DATA_t * residualCouplingBlockData = (RESIDUAL_COUPLING_BLOCK_DATA_t *)priv_;

    for(int32_t i =0; i<residualCouplingBlockData->n_flows_; i++)
    {
        delete residualCouplingBlockData->residualCouplingLayerList_[i];
    }

    free(residualCouplingBlockData->residualCouplingLayerList_);
    delete residualCouplingBlockData;
}

MatrixXf ResidualCouplingBlock::forward(const MatrixXf & x, const MatrixXf & g)
{
    RESIDUAL_COUPLING_BLOCK_DATA_t * residualCouplingBlockData = (RESIDUAL_COUPLING_BLOCK_DATA_t *)priv_;
    
    MatrixXf xx = x;
    for(int32_t i = residualCouplingBlockData->n_flows_-1; i>=0; i--)
    {
        xx = nn_flip(xx,1);
        xx = residualCouplingBlockData->residualCouplingLayerList_[i]->forward(xx,g);
    }

    return xx;
}
