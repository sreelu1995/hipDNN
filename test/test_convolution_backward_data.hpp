#ifndef TEST_CONVOLUTION_BACKWARD_COMMON_HPP
#define TEST_CONVOLUTION_BACKWARD_COMMON_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_conv_backward_data(convulution_Size &c, dataType *src,
                             dataType *weights, dataType *grad, dataType *bias, dataType *dst, float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih, c.iw));

  hipdnnFilterDescriptor_t filt_desc;
  checkHIPDNN(hipdnnCreateFilterDescriptor(&filt_desc));
  int filterDimA[] = {c.oc, c.ic, c.kh, c.kw};
  checkHIPDNN(hipdnnSetFilterNdDescriptor(filt_desc, HIPDNN_DATA_FLOAT,
                                          HIPDNN_TENSOR_NCHW, 4, filterDimA));
  hipdnnConvolutionDescriptor_t conv_desc;
  checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv_desc));
  checkHIPDNN(hipdnnSetConvolution2dDescriptor(
      conv_desc, c.padh, c.padw, c.strh, c.strw, c.dilh, c.dilw,
      HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

  checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(
      conv_desc, in_desc, filt_desc, &c.mb, &c.oc, &c.oh, &c.ow));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.oc, c.oh, c.ow));
  hipdnnConvolutionFwdAlgo_t algo;
  int MaxAlgoCount = 5;
  size_t ws_size;
  float *ws_data;
  int calgo;
  hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount];

  hipdnnFindConvolutionForwardAlgorithmEx(
      hipdnn, in_desc, src, filt_desc, weights, conv_desc, out_desc, dst,
      MaxAlgoCount, &calgo, algoPerf, ws_data, ws_size);
  algo = (hipdnnConvolutionFwdAlgo_t)algoPerf[0].algo;

  checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(
      hipdnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  
  hipMalloc(&ws_data, ws_size);

  float alpha = 1.f;
  float beta = 0.f;

  checkHIPDNN(hipdnnConvolutionForward(hipdnn, &alpha, in_desc, src, filt_desc,
                                       weights, conv_desc, algo, ws_data,
                                       ws_size, &beta, out_desc, dst));


     hipdnnConvolutionBwdDataAlgo_t algo_bd; 
     ws_size =0;
     hipdnnConvolutionBwdDataAlgoPerf_t algoPerf_bd[MaxAlgoCount];        

     hipdnnFindConvolutionBackwardDataAlgorithmEx(hipdnn, filt_desc, weights, in_desc, src, conv_desc, out_desc, dst, MaxAlgoCount , &calgo, algoPerf_bd, ws_data, ws_size);

 algo_bd = algoPerf_bd[0].algo;
   
  checkHIPDNN(hipdnnGetConvolutionBackwardDataWorkspaceSize(hipdnn, filt_desc, in_desc, conv_desc, out_desc, algo_bd, &ws_size));

  hipMalloc(&ws_data, ws_size);

    high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

       timer.restart();
       
       checkHIPDNN(hipdnnConvolutionBackwardData( hipdnn, &alpha, filt_desc, weights,
                                 out_desc, dst, conv_desc, algo_bd, ws_data, ws_size,
                                 &beta, out_desc, grad));  

       hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
	}
	
	*avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);
  
  hipFree(ws_data);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyFilterDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}

#endif // TEST_CONVOLUTION_FORWARD_COMMON_HPP 
