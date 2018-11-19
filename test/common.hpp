#ifndef COMMON_HPP
#define COMMON_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"

Desc calculate_Dims(Desc inputDesc, Desc filterDesc, int pad[2],
                               int stride[2], int dilution[2]);

__global__ void dev_const(hipLaunchParm lp, float *px, float k);

struct convulution_Size {
  convulution_Size(int mb, int ng, int ic, int ih, int iw, int oc,
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

struct activation_params_t {
  int n, channels, height, width;
  activation_params_t(int n, int channels, int height, int width)
      : n(n), channels(channels), height(height), width(width) {}
};

struct BNorm_params_t {
  int mb, ic, ih, iw;
  BNorm_params_t(int mb, int ic, int ih, int iw)
      : mb(mb), ic(ic), ih(ih), iw(iw) {}
};

struct LRN_params_t {
  int mb, ic, ih, iw;
  LRN_params_t(int mb, int ic, int ih, int iw)
      : mb(mb), ic(ic), ih(ih), iw(iw) {}
};

struct pool_bwd {
  size_t in, ichannel, iheight, iwidth;
  size_t wheight, wwidth;
  size_t vpadding, hpadding;
  size_t vstride, hstride;
  int on, ochannel, oheight, owidth;

  pool_bwd(size_t in, size_t ichannel, size_t iheight, size_t iwidth,
                 size_t wheight, size_t wwidth, size_t vpadding,
                 size_t hpadding, size_t vstride, size_t hstride)
      : in(in), ichannel(ichannel), iheight(iheight), iwidth(iwidth),
        wheight(wheight), wwidth(wwidth), vpadding(vpadding),
        hpadding(hpadding), vstride(vstride), hstride(hstride) {}

  pool_bwd(size_t in, size_t ichannel, size_t iheight, size_t iwidth,
                 size_t wheight, size_t wwidth, size_t vpadding,
                 size_t hpadding, size_t vstride, size_t hstride, size_t on,
                 size_t ochannel, size_t oheight, size_t owidth)
      : in(in), ichannel(ichannel), iheight(iheight), iwidth(iwidth),
        wheight(wheight), wwidth(wwidth), vpadding(vpadding),
        hpadding(hpadding), vstride(vstride), hstride(hstride), on(on),
        ochannel(ochannel), oheight(oheight), owidth(owidth) {}
};

#endif //COMMON_HPP
