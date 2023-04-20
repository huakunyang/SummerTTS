#include "WN.h"
#include "tts_logger.h"
#include "nn_conv1d.h"
#include "nn_sigmoid.h"
#include "nn_tanh.h"

typedef struct
{
    int32_t isMS_;
    int32_t n_layers_;
    int32_t hidden_channels_;
    int32_t kSize_;
    nn_conv1d ** in_layers_;
    nn_conv1d ** res_skip_layers_;
    nn_conv1d * cond_layer_;

}WN_DATA_t;

WN::WN(float * modelData, int32_t & offset, int32_t dilation_rate, int32_t isMS)
{
    WN_DATA_t * wnData = new WN_DATA_t();
    if(NULL == wnData)
    {
        tts_log(TTS_LOG_ERROR, "WN: Failed to allocate memory for internal data block\n");
        return;
    }

    memset(wnData,0,sizeof(WN_DATA_t));

    int32_t curOffset = offset;
    wnData->isMS_ = isMS;
    wnData->n_layers_ = (int32_t)modelData[curOffset++];
    wnData->kSize_ = (int32_t)modelData[curOffset++];
    wnData->in_layers_ = (nn_conv1d **)(malloc(sizeof(nn_conv1d *)*wnData->n_layers_));
    
    int32_t dilation = dilation_rate;
    for(int32_t i = 0; i<wnData->n_layers_; i++)
    {
        dilation = dilation*dilation_rate;
        int32_t padding = (int32_t)((wnData->kSize_ * dilation - dilation) / 2);

        wnData->in_layers_[i] = new nn_conv1d(modelData,curOffset,padding,dilation,0);        
    }
    
    if(wnData->n_layers_ >0)
    {
        wnData->hidden_channels_ = wnData->in_layers_[0]->get_in_channels_num();
    }

    wnData->res_skip_layers_ = (nn_conv1d **)(malloc(sizeof(nn_conv1d *)*wnData->n_layers_));

    for(int32_t i = 0; i<wnData->n_layers_; i++)
    {
        wnData->res_skip_layers_[i] = new nn_conv1d(modelData,curOffset);        
    }
    
    if(wnData->isMS_ == 1)
    {
        wnData->cond_layer_ = new nn_conv1d(modelData,curOffset);
    }

    offset = curOffset;
    priv_ = (void*)wnData;
}

WN::~WN()
{
    WN_DATA_t * wnData = (WN_DATA_t *)priv_;
    for(int32_t i = 0; i<wnData->n_layers_; i++)
    {
        delete wnData->in_layers_[i];
        delete wnData->res_skip_layers_[i];
    }

    if(wnData->isMS_ == 1)
    {
        delete wnData->cond_layer_;
    }

    free(wnData->in_layers_);
    free(wnData->res_skip_layers_);
    delete wnData;
}

MatrixXf fused_add_tanh_sigmoid_multiply(const MatrixXf & input_a, const MatrixXf & input_b, int32_t n_channels)
{
    MatrixXf in_act = input_a.rowwise() + input_b.row(0);

    MatrixXf in_act_1 = in_act.block(0,0,in_act.rows(),n_channels);
    MatrixXf in_act_2 = in_act.block(0,n_channels,in_act.rows(),in_act.cols() - n_channels);

    MatrixXf t_act = nn_tanh(in_act_1);
    MatrixXf s_act = nn_sigmoid(in_act_2);

    MatrixXf acts = t_act.array()*s_act.array();

    return acts;
}

MatrixXf WN::forward(const MatrixXf & x, const MatrixXf & g)
{
    
    WN_DATA_t * wnData = (WN_DATA_t *)priv_;
    
    MatrixXf xx = x;
    MatrixXf output = MatrixXf::Zero(xx.rows(),xx.cols());

    MatrixXf gg;
    if(wnData->isMS_ == 1)
    {
        gg = wnData->cond_layer_->forward(g);
    }

    for(int32_t i = 0; i< wnData->n_layers_; i++)
    {
        MatrixXf x_in = wnData->in_layers_[i]->forward(xx);
        
        MatrixXf g_l;
        if(wnData->isMS_ == 1)
        {
            int32_t cond_offset = i * 2 * wnData->hidden_channels_; 
            g_l = gg.block(0,cond_offset,1,2 * wnData->hidden_channels_);
        }
        else
        {
            g_l = MatrixXf::Zero(x_in.rows(), x_in.cols());
        }

        MatrixXf acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, wnData->hidden_channels_);

        MatrixXf res_skip_acts = wnData->res_skip_layers_[i]->forward(acts);

        if(i < wnData->n_layers_-1)
        {
            MatrixXf res_acts = res_skip_acts.block(0,0,res_skip_acts.rows(),wnData->hidden_channels_);
            xx = xx+res_acts;
            MatrixXf res_acts2 = res_skip_acts.block(0,wnData->hidden_channels_,
                                                     res_skip_acts.rows(),res_skip_acts.cols()-wnData->hidden_channels_);
            
            output = output + res_acts2;
        }
        else
        {
            output = output + res_skip_acts;
        }
    }

    return output;
}

