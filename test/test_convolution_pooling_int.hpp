#ifndef TEST_CONVOLUTION_POOLING_HPP
#define TEST_CONVOLUTION_POOLING_HPP

#include "hipDNN.h"
#include "hipDNN_test_common.h"
#include "gtest/gtest.h"

struct test_conv_sizes {
  test_conv_sizes(int mb, int ng, int ic, int ih, int iw, int oc,
                           int oh, int ow, int kh, int kw, int padh, int padw,
                           int strh, int strw, int dilh = 0, int dilw = 0)
      : mb(mb), ng(ng), ic(ic), ih(ih), iw(iw), oc(oc), oh(oh), ow(ow), kh(kh),
        kw(kw), padh(padh), padw(padw), strh(strh), strw(strw), dilh(dilh),
        dilw(dilw) {}
  int mb;         // mini batches
  int ng;         // number of groups
  int ic, ih, iw; // Input channels, height and width
  int oc, oh, ow; // Output channels, height and width
  int kh, kw;     // kernel height and width
  int padh, padw; // padding along height and width
  int strh, strw; // stride along height and width
  int dilh, dilw; // dilation along height and width
};


struct test_pooling_descriptor {
  int mb, c;      // Minibatch and channels
  int ih, iw;     // input dimensions
  int oh, ow;     // output dimensions
  int kh, kw;     // kernel dimensions
  int padt, padl; // padding dimensions
  int strh, strw; // stride dimensions
  test_pooling_descriptor(int mb, int c, int ih, int iw, int oh, int ow, int kh,
                     int kw, int padt, int padl, int strh, int strw)
      : mb(mb), c(c), ih(ih), iw(iw), oh(oh), ow(ow), kh(kh), kw(kw),
        padt(padt), padl(padl), strh(strh), strw(strw) {}
};

// Note we are currently only dealing with 2D convolution
Desc calculateConvDesc(Desc inputDesc, Desc filterDesc, int pad[2],
                               int stride[2]) {
  assert(inputDesc.C == filterDesc.C);
  int outputHeight =
      ((inputDesc.H - filterDesc.H + 2 * pad[0]) / stride[0]) + 1;
  int outputWidth = ((inputDesc.W - filterDesc.W + 2 * pad[1]) / stride[1]) + 1;
  Desc outputDesc(inputDesc.N, filterDesc.N, outputHeight, outputWidth);
  return outputDesc;
}

template <typename dataType>
void compute_conv_forward(test_conv_sizes &c, dataType *src,
                             dataType *weights, dataType *bias, dataType *dst, float *avg_time) {

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
  int MaxAlgoCount = 1;
  size_t ws_size{0};
  float *ws_data{nullptr};
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

    high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        checkHIPDNN(hipdnnConvolutionForward(hipdnn, &alpha, in_desc, src, filt_desc,
                                       weights, conv_desc, algo, ws_data,
                                       ws_size, &beta, out_desc, dst));
     
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

template <typename dataType>
void compute_pooling_fwd(test_pooling_descriptor &c, dataType *src,
                                dataType *dst, float *avg_time) {

  hipdnnHandle_t handle;
  checkHIPDNN(hipdnnCreate(&handle));
  hipdnnTensorDescriptor_t in_desc, out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.c, c.ih, c.iw));

  hipdnnPoolingDescriptor_t pool_desc;
  checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));
  checkHIPDNN(hipdnnSetPooling2dDescriptor(pool_desc, HIPDNN_POOLING_MAX,
                                           HIPDNN_NOT_PROPAGATE_NAN, c.kw, c.kh,
                                           c.padt, c.padl, c.strh, c.strw));
  checkHIPDNN(hipdnnGetPooling2dForwardOutputDim(pool_desc, in_desc, &c.mb,
                                                 &c.c, &c.oh, &c.ow));

  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.c, c.oh, c.ow));
  float alpha = 1.f;
  float beta = 0.f;

  high_resolution_timer_t timer;
  std::vector<double> time_vector(benchmark_iterations, 0);
  for (int i = 0; i < benchmark_iterations; i++) {

      timer.restart();
      checkHIPDNN(hipdnnPoolingForward(handle, pool_desc, &alpha, in_desc, src,
                                   &beta, out_desc, dst));
      
       hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

    *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);

  checkHIPDNN(hipdnnDestroyTensorDescriptor(in_desc));
  checkHIPDNN(hipdnnDestroyTensorDescriptor(out_desc));
  checkHIPDNN(hipdnnDestroyPoolingDescriptor(pool_desc));
  checkHIPDNN(hipdnnDestroy(handle));
}

#endif //TEST_CONVOLUTION_POOLING_HPP
