#include "ConvFlow.h"
#include "nn_conv1d.h"
#include "DDSConv.h"
#include "tts_logger.h"
#include "nn_softmax.h"
#include "nn_cumsum.h"
#include "nn_softplus.h"

#define PAD_CONST  (0.5397424172369522)
#define DEFAULT_MIN_BIN_WIDTH (1e-3)
#define DEFAULT_MIN_BIN_HEIGHT (1e-3)
#define DEFAULT_MIN_DERIVATIVE (1e-3)

typedef struct
{
    int32_t numBins_;
    int32_t filterChan_;
    nn_conv1d * pre_;
    DDSConv * convs_;
    nn_conv1d * proj_;

}CONV_FLOW_DATA_t;

ConvFlow::ConvFlow(float * modelData, int32_t & offset)
{
    CONV_FLOW_DATA_t * convFlowData = new CONV_FLOW_DATA_t();
    if(NULL == convFlowData)
    {
        tts_log(TTS_LOG_ERROR, "ConvFlow: Failed to allocate memory for internal data block\n");
        return;
    }
    
    memset(convFlowData,0,sizeof(CONV_FLOW_DATA_t));
    
    int32_t curOffset = offset;
    
    convFlowData->numBins_ = 10;
    convFlowData->pre_ = new nn_conv1d(modelData,curOffset);
    convFlowData->convs_ = new DDSConv(modelData,curOffset);
    convFlowData->proj_ = new nn_conv1d(modelData,curOffset);
    convFlowData->filterChan_ = convFlowData->pre_->get_out_channels_num();

    offset = curOffset;
    priv_ = (void *)convFlowData;
}

ConvFlow::~ConvFlow()
{
    CONV_FLOW_DATA_t * convFlowData = (CONV_FLOW_DATA_t *)priv_;

    delete convFlowData->pre_;
    delete convFlowData->convs_;
    delete convFlowData->proj_;
    delete convFlowData;
}

MatrixXf searchsorted(const MatrixXf & a, const MatrixXf & b)
{
    MatrixXf bin_locations = a;
    bin_locations.col(a.cols()-1) = a.col(a.cols()-1).array() + 1e-6;

    MatrixXf ret = MatrixXf::Zero(a.rows(),1);

    for(int32_t i = 0; i< a.rows(); i++)
    {
        int32_t sum = 0;
        for(int32_t j = 0; j< a.cols(); j++)
        {
            if(b(i,0) >= bin_locations(i,j))
            {
                sum = sum + 1;
            }
        }
        ret(i,0) = sum-1;
    }

    return ret;
}

MatrixXf unconstrained_rational_quadratic_spline(const MatrixXf & x,
                                                 const MatrixXf & unnormalized_widths,
                                                 const MatrixXf & unnormalized_heights,
                                                 const MatrixXf & unnormalized_derivatives,
                                                 float tail_bound,
                                                 MatrixXf & outlogabsdet)
{
    auto X_M0 = x.array() < tail_bound;
    auto X_M1 = x.array() > -tail_bound;
    auto inside_interval_mask = X_M0.array()* X_M1.array();
    auto outside_interval_mask = inside_interval_mask.array()-1;

    MatrixXf outputs   = MatrixXf::Zero(x.rows(), x.cols());
    MatrixXf logabsdet = MatrixXf::Zero(x.rows(), x.cols());

    MatrixXf unnormalized_derivatives_pad = MatrixXf::Zero(unnormalized_derivatives.rows(),unnormalized_derivatives.cols()+2);
    unnormalized_derivatives_pad.block(0,1,unnormalized_derivatives.rows(),unnormalized_derivatives.cols()) = unnormalized_derivatives;
    
    unnormalized_derivatives_pad.col(0).array() = PAD_CONST;
    unnormalized_derivatives_pad.col(unnormalized_derivatives_pad.cols()-1).array() = PAD_CONST;

    for(int32_t i = 0; i< outside_interval_mask.rows(); i++)
    {
        for(int32_t j = 0; j< outside_interval_mask.cols(); j++)
        {
            if(outside_interval_mask(i,j) > 0.5)
            {
                outputs(i,j) = x(i,j);
                logabsdet(i,j) = 0.0;
            }
        }
    }

    MatrixXf unnormalized_widths_masked = MatrixXf::Zero(unnormalized_widths.rows(),unnormalized_widths.cols());
    MatrixXf unnormalized_heights_masked = MatrixXf::Zero(unnormalized_heights.rows(),unnormalized_heights.cols());
    MatrixXf unnormalized_derivatives_masked = MatrixXf::Zero(unnormalized_derivatives.rows(),unnormalized_derivatives.cols());

    MatrixXf input_inter_masked = MatrixXf::Zero(x.rows(),x.cols());
    for(int32_t i = 0; i< inside_interval_mask.rows(); i++)
    {
        for(int32_t j = 0; j< inside_interval_mask.cols(); j++)
        {
            if(inside_interval_mask(i,j) > 0.5)
            {
                input_inter_masked(i,j) = x(i,j);
                
            }
        }

        if(inside_interval_mask(i,0) > 0.5)
        {
            unnormalized_widths_masked.row(i) = unnormalized_widths.row(i);
            unnormalized_heights_masked.row(i) = unnormalized_heights.row(i);
            unnormalized_derivatives_masked.row(i) = unnormalized_derivatives.row(i);
        }
    }

    int32_t num_bins = unnormalized_widths_masked.cols();
    MatrixXf widths = nn_softmax(unnormalized_widths_masked,0); 

    widths = widths.array()*(1 - DEFAULT_MIN_BIN_WIDTH * num_bins) + DEFAULT_MIN_BIN_WIDTH;
    MatrixXf cumwidths = nn_cumsum(widths,1);
    MatrixXf cumwidths_padded = MatrixXf::Zero(cumwidths.rows(),cumwidths.cols()+1);
    cumwidths_padded.block(0,1,cumwidths.rows(),cumwidths.cols()) = cumwidths;

    float right  = tail_bound;
    float left   = -tail_bound;
    float bottom = -tail_bound;
    float top    = tail_bound;

    MatrixXf cumwidths_added = cumwidths_padded.array()*(right-left) + left;
    cumwidths_added.col(0).array() = left;
    cumwidths_added.col(cumwidths_added.cols()-1).array() = right;

    MatrixXf width_subs = cumwidths_added.block(0,1,cumwidths.rows(),cumwidths.cols()) - 
                          cumwidths_added.block(0,0,cumwidths.rows(),cumwidths.cols());

    MatrixXf derivatives_softplus = (nn_softplus(unnormalized_derivatives_pad)).array() + DEFAULT_MIN_DERIVATIVE;

    MatrixXf height_softmax = nn_softmax(unnormalized_heights_masked,0);

    MatrixXf height_added = height_softmax.array()*(1 - DEFAULT_MIN_BIN_HEIGHT * num_bins) + DEFAULT_MIN_BIN_HEIGHT;
    MatrixXf height_added_cumsum = nn_cumsum(height_added,1);
    MatrixXf height_added_cumsum_padded = MatrixXf::Zero(height_added_cumsum.rows(),1+height_added_cumsum.cols());
    height_added_cumsum_padded.block(0,1,height_added_cumsum.rows(),height_added_cumsum.cols()) = height_added_cumsum;

    height_added = height_added_cumsum_padded.array()*(top - bottom) + bottom;
    
    height_added.col(0).array() = bottom;
    height_added.col(height_added.cols()-1).array() = top;

    MatrixXf heights;
    heights = height_added.block(0,1,height_added.rows(), height_added.cols()-1) - height_added.block(0,0,height_added.rows(), height_added.cols()-1);

    MatrixXf bin_idx = searchsorted(height_added, x);

    MatrixXf delta = heights.array()/width_subs.array();

    MatrixXf input_cumwidths = MatrixXf::Zero(bin_idx.rows(),1);
    MatrixXf input_bin_widths = MatrixXf::Zero(bin_idx.rows(),1);
    MatrixXf input_cumheights = MatrixXf::Zero(bin_idx.rows(),1);
    MatrixXf input_delta = MatrixXf::Zero(bin_idx.rows(),1);
    MatrixXf input_derivatives = MatrixXf::Zero(bin_idx.rows(),1);
    MatrixXf input_derivatives_plus_one = MatrixXf::Zero(bin_idx.rows(),1);
    MatrixXf input_heights = MatrixXf::Zero(bin_idx.rows(),1);

    for(int32_t i = 0; i<bin_idx.rows();i++)
    {
        int32_t index = (int32_t)bin_idx(i,0);
        input_cumwidths(i,0) = cumwidths_added(i,index); 
        input_bin_widths(i,0) = width_subs(i,index); 
        input_cumheights(i,0) = height_added(i,index);
        input_delta(i,0) = delta(i,index);
        input_derivatives(i,0) = derivatives_softplus(i,index);  
        input_derivatives_plus_one(i,0) = (derivatives_softplus.block(0,1,
                                                                    derivatives_softplus.rows(),
                                                                    derivatives_softplus.cols()-1))(i,index);  
        input_heights(i,0) = heights(i,index);  
    }
    
    MatrixXf a = (((x - input_cumheights).array() * 
          (input_derivatives + input_derivatives_plus_one - input_delta*2).array()
          + input_heights.array() * (input_delta - input_derivatives).array()));

    MatrixXf b =  ((input_heights.array() * input_derivatives.array())
                 - (x - input_cumheights).array() * 
                 (input_derivatives + input_derivatives_plus_one - 2 * input_delta).array());

    MatrixXf c = - (input_delta.array() * (x - input_cumheights).array());
    MatrixXf discriminant = b.array().pow(2) - a.array() * c.array()*4;

    MatrixXf root = (c.array()*2) / (-b.array() - discriminant.array().sqrt());

    MatrixXf outputs_tmp = root.array() * input_bin_widths.array() + input_cumwidths.array();

    MatrixXf theta_one_minus_theta = root.array() * (1 - root.array());
    MatrixXf denominator = input_delta.array() + ((input_derivatives + 
                                          input_derivatives_plus_one - input_delta*2).array()
                                          * theta_one_minus_theta.array());
    MatrixXf derivative_numerator = input_delta.array().pow(2) * 
                                    (input_derivatives_plus_one.array() * root.array().pow(2)+
                                    input_delta.array() * theta_one_minus_theta.array()*2 +
                                    input_derivatives.array() * (1 - root.array()).pow(2));
    
    MatrixXf logabsdet_tmp = -(derivative_numerator.array().log() - denominator.array().log()*2);
    
    for(int32_t i = 0; i< inside_interval_mask.rows(); i++)
    {
        for(int32_t j = 0; j<inside_interval_mask.cols(); j++)
        {
            if(inside_interval_mask(i,j) > 0.5)
            {
                outputs(i,j) = outputs_tmp(i,j);
                logabsdet(i,j) = logabsdet_tmp(i,j);
            }
        }
    }

    outlogabsdet = logabsdet;
    return outputs;
}

MatrixXf ConvFlow::forward(const MatrixXf & x, const MatrixXf & g)
{
    CONV_FLOW_DATA_t * convFlowData = (CONV_FLOW_DATA_t *)priv_;

    MatrixXf x0 = (x.block(0,0,(int32_t)(x.rows()/2),x.cols())).transpose();
    MatrixXf x1 = (x.block((int32_t)(x.rows()/2),0,(int32_t)(x.rows()/2),x.cols())).transpose();

    MatrixXf h = convFlowData->pre_->forward(x0);
    h = convFlowData->convs_->forward(h,g,1);
    h = convFlowData->proj_->forward(h);

    MatrixXf filterSqrtMat = MatrixXf::Zero(1,1);
    filterSqrtMat(0,0) = convFlowData->filterChan_;
    float filterSqrt = (filterSqrtMat.array().sqrt())(0,0);
    
    MatrixXf unnormalized_widths = (h.block(0,0,h.rows(),convFlowData->numBins_))/filterSqrt;
    MatrixXf unnormalized_heights = (h.block(0,convFlowData->numBins_,h.rows(),convFlowData->numBins_))/filterSqrt;
    MatrixXf unnormalized_derivatives = h.block(0,2*convFlowData->numBins_, h.rows(), h.cols()-2*convFlowData->numBins_);

    MatrixXf logabsdet;
    MatrixXf outputs = unconstrained_rational_quadratic_spline(x1,unnormalized_widths,unnormalized_heights,unnormalized_derivatives,5.0,logabsdet); 

    MatrixXf result = MatrixXf::Zero(x0.rows(),x0.cols()+outputs.cols());

    result.block(0,0,x0.rows(),x0.cols()) = x0;
    result.block(0,x0.cols(),outputs.rows(),outputs.cols()) = outputs;

    return result;
}
