#ifndef TEST_CONVOLUTION_FORWARD_COMMON_HPP
#define TEST_CONVOLUTION_FORWARD_COMMON_HPP

#include "gtest/gtest.h"
#include "hipDNN_test_common.h"
#include "hipDNN.h"


// CPU reference code
// Input buffers are all host pointers
template<typename dataType>
void compute_cpuref_conv_fwd(test_convolution_sizes_t& c, dataType* src, dataType* weights, dataType* bias, dataType* dst) {

    for (int n = 0; n < c.mb; n++) {
        for (int oc = 0; oc < c.oc; oc++) {
            for (int oh = 0; oh < c.oh; oh++) {
                for (int ow = 0; ow < c.ow; ow++) {
                    dataType a = 0; // For accumulation
                    for (int ic = 0; ic < c.ic; ic++) {
                        for (int kh = 0; kh < c.kh; kh++) {
                            for (int kw = 0; kw < c.kw; kw++) {
                                int iw = ow * c.strw
                                      - c.padw + kw * (1 + c.dilw);
                                int ih = oh * c.strh
                                      - c.padh + kh * (1 + c.dilh);
                                if (iw < 0 || iw >= c.iw) continue;
                                if (ih < 0 || ih >= c.ih) continue;
                                size_t iidx = n * c.ic * c.ih * c.iw
                                    + ic * c.ih * c.iw + ih * c.iw + iw;
                                size_t widx = oc * c.ic  * c.kh * c.kw
                                    + ic * c.kh * c.kw + kh * c.kw + kw;
                                a += (dataType)src[iidx] *  weights[widx];
                            }
                        }
                    }

                    float a_fp = (float)a;

                    a_fp += (float)(bias ?
                        bias[oc] :
                        0);

                    size_t oidx = n * c.oc * c.oh * c.ow
                             + oc * c.oh * c.ow + oh * c.ow + ow;
                    dst[oidx] = (dataType)a_fp;
                }
            }
        }
    }
}


// GPU Reference code
template <typename dataType>
void convolution_forward(test_convolution_sizes_t& c, dataType* src, dataType* weights, dataType* bias, dataType* dst) {


  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        c.mb, c.ic, c.ih, c.iw));

  /* float *src;
  hipMalloc(
       &src, in_n * in_c * in_h * in_w * sizeof(float));*/


  hipdnnTensorDescriptor_t filt_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&filt_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        filt_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        c.oc, c.ic, c.kh, c.kw));

 /* float *weights;
  hipMalloc(
      &weights, filt_k * filt_c * filt_h * filt_w * sizeof(float));*/

  hipdnnConvolutionDescriptor_t conv_desc;
  checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv_desc));
  checkHIPDNN(hipdnnSetConvolution2dDescriptor(
        conv_desc,
        c.padh, c.padw, c.strh, c.strw, c.dilh, c.dilw,
        HIPDNN_CONVOLUTION, HIPDNN_DATA_FLOAT));

  checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &c.mb, &c.oc, &c.oh, &c.ow));


  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        c.mb, c.oc, c.oh, c.ow));

  /* float *dst;
  hipMalloc(
        &dst, out_n * out_c * out_h * out_w * sizeof(float));*/

    hipdnnConvolutionFwdAlgo_t algo;
    int MaxAlgoCount =1;
    size_t ws_size;
    float *ws_data;
    int calgo;
    hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount];

  hipdnnFindConvolutionForwardAlgorithmEx(hipdnn, in_desc, src, filt_desc, weights, conv_desc, out_desc, dst, MaxAlgoCount , &calgo, algoPerf, ws_data, ws_size);
  algo = algoPerf[0].algo;


  checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(hipdnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  hipMalloc(&ws_data, ws_size);

  // perform
  float alpha = 1.f;
  float beta = 0.f;

  checkHIPDNN(hipdnnConvolutionForward(
      hipdnn,
      &alpha, in_desc, src, filt_desc, weights,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, dst));


  // finalizing
  hipFree(ws_data);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyFilterDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}



#endif //TEST_CONVOLUTION_FORWARD_COMMON_HPP
