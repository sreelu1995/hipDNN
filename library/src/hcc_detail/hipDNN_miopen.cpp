/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hipDNN.h>
#include <hcc_detail/hipDNN_miopen.h>
#include <stdint.h>
#include <map>

#ifndef thread_local
#if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
#define thread_local _Thread_local
#elif defined _WIN32 && ( \
      defined _MSC_VER || \
      defined __ICL || \
      defined __DMC__ || \
      defined __BORLANDC__ )
#define thread_local __declspec(thread) 
/* note that ICC (linux) and Clang are covered by __GNUC__ */
#elif defined __GNUC__ || \
       defined __SUNPRO_C || \
       defined __xlC__
#define thread_local __thread
#else
//#error "Cannot define thread_local"
// NOT THREAD SAFE.
#define thread_local
#endif
#endif

//HGSOS not implemented yet because i dont know how to get the device pointers from the desc!!!
static thread_local void* sConvolutionForwardAlgorithmWorkspace;

//Eventually those can be merged, but currently i will not mix device pointers with host data in the same map...
//HGSOS static std::map<miopenPoolingDescriptor_t, void *>  sPoolingDescToWorkspace;  ????
static std::map<miopenTensorDescriptor_t, int8_t*>  sDescToWorkspacePooling;  //device pointers
static std::map<miopenTensorDescriptor_t, size_t>  sDescToWorkspacePoolingSize;  //host


static std::map<miopenTensorDescriptor_t, int8_t*>  sDescToWorkspaceLRN;  //device pointers
static std::map<miopenTensorDescriptor_t, size_t>  sDescToWorkspaceLRNSize;  //host

//=============================================================================



hipdnnStatus_t miopenTohipdnnStatus(miopenStatus_t  cStatus)
{
    hipdnnStatus_t retVal;
    switch(cStatus)
    {
    case miopenStatusSuccess:   
        retVal = HIPDNN_STATUS_SUCCESS;
        break;
    case miopenStatusNotInitialized:  
        retVal = HIPDNN_STATUS_NOT_INITIALIZED;
        break;  
    case miopenStatusAllocFailed:
        retVal = HIPDNN_STATUS_ALLOC_FAILED;
        break;  
    case miopenStatusBadParm:
        retVal = HIPDNN_STATUS_BAD_PARAM;
        break;  
    case miopenStatusInternalError:
        retVal = HIPDNN_STATUS_INTERNAL_ERROR;
        break;  
    case miopenStatusInvalidValue:
        retVal = HIPDNN_STATUS_INVALID_VALUE;
        break;      
    case miopenStatusUnknownError:
        retVal = HIPDNN_STATUS_EXECUTION_FAILED;
        break;
    case miopenStatusNotImplemented:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    default:
        retVal = HIPDNN_STATUS_INTERNAL_ERROR;
    }
    return retVal;
} 

//=============================================================================

hipdnnStatus_t hipTomiopenDataType(hipdnnDataType_t in, miopenDataType_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    switch(in)
    {
    case HIPDNN_DATA_FLOAT:
        *out = miopenFloat;
        retVal = HIPDNN_STATUS_SUCCESS;
        break;
    case HIPDNN_DATA_HALF:
        *out = miopenHalf;
        retVal = HIPDNN_STATUS_SUCCESS;
        break;
    case HIPDNN_DATA_DOUBLE:
    case HIPDNN_DATA_INT8:
    case HIPDNN_DATA_INT32:
    case HIPDNN_DATA_INT8x4:
        break;
    }
    
    return retVal;
}

hipdnnStatus_t miopenTohipDataType(miopenDataType_t in, hipdnnDataType_t* out)
{
    switch(in)
    {
    case miopenFloat: 
        *out = HIPDNN_DATA_FLOAT;
        break;
    case miopenHalf:
        *out = HIPDNN_DATA_HALF;
        break;
    }
    
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t miopenTohipOpTensorOp(miopenTensorOp_t in, hipdnnOpTensorOp_t* out)
{
    switch(in)
    {
    case miopenTensorOpAdd:
        *out = HIPDNN_OP_TENSOR_ADD;
        break;
    case miopenTensorOpMul:
        *out = HIPDNN_OP_TENSOR_MUL;
        break;
    case miopenTensorOpMin:
        *out = HIPDNN_OP_TENSOR_MIN;
        break;
    case miopenTensorOpMax:
        *out = HIPDNN_OP_TENSOR_MAX;
        break;
    }
    
    return HIPDNN_STATUS_SUCCESS;
}

    
hipdnnStatus_t hipTomiopenOpTensorOp(hipdnnOpTensorOp_t in, miopenTensorOp_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    
    switch(in)
    {
    case HIPDNN_OP_TENSOR_ADD :
        *out = miopenTensorOpAdd;
        break;
    case HIPDNN_OP_TENSOR_MUL :
        *out = miopenTensorOpMul;
        break;
    case HIPDNN_OP_TENSOR_MIN:
        *out = miopenTensorOpMin;
        break;
    case HIPDNN_OP_TENSOR_MAX:
        *out = miopenTensorOpMax;
        break;
    case HIPDNN_OP_TENSOR_SQRT:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    
    return retVal;
}


//=============================================================================

hipdnnConvolutionMode_t miopenTohipConvolutionMode( miopenConvolutionMode_t in )
{
    if( in == miopenConvolution )
        return HIPDNN_CONVOLUTION;
    /*else if( in == miopenCrossCorrelation )
        return HIPDNN_CROSS_CORRELATION;*/ //TODO: to be added
    
    return HIPDNN_CONVOLUTION;
}

miopenConvolutionMode_t hipTomiopenConvolutionMode( hipdnnConvolutionMode_t in )
{
    if( in == HIPDNN_CONVOLUTION )
        return miopenConvolution;
    /*else if( in == HIPDNN_CROSS_CORRELATION )
        return miopenCrossCorrelation;*/ //TODO: to be added
    
    return miopenConvolution;
}



//=============================================================================

hipdnnStatus_t  hipTomiopenPoolingMode( hipdnnPoolingMode_t in, 
                                        miopenPoolingMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_POOLING_MAX:
        *out = miopenPoolingMax;
        break;
    case HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
        *out = miopenPoolingAverage;
        break;
    case HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
        *out = miopenPoolingAverage;
        break;
    case HIPDNN_POOLING_MAX_DETERMINISTIC:
        *out = miopenPoolingMax;
        break;
    }
    return retVal;
} 

hipdnnStatus_t  miopenTohipPoolingMode( miopenPoolingMode_t in, 
                                        hipdnnPoolingMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case miopenPoolingMax:
        *out = HIPDNN_POOLING_MAX;
        break;
    case miopenPoolingAverage :
        *out = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
//HGSOS     *out = HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
//HGSOS     *out = HIPDNN_POOLING_MAX_DETERMINISTIC;
    default:
	retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
} 


//=============================================================================

hipdnnStatus_t  hipTomiopenLRNMode( hipdnnLRNMode_t in, 
                                    miopenLRNMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_LRN_WITHIN_CHANNEL:
        *out = miopenLRNWithinChannel;
        break;
    case HIPDNN_LRN_CROSS_CHANNEL_DIM1:
        *out = miopenLRNCrossChannel;
        break;
    }
    return retVal;
} 

hipdnnStatus_t  miopenTohipLRNMode( miopenLRNMode_t  in, 
                                    hipdnnLRNMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case miopenLRNWithinChannel :
        *out = HIPDNN_LRN_WITHIN_CHANNEL;
        break;
    case miopenLRNCrossChannel :
        *out = HIPDNN_LRN_CROSS_CHANNEL_DIM1;
        break;
    }
    return retVal;
} 

//=============================================================================


miopenBatchNormMode_t hipTomiopenBatchNormMode( hipdnnBatchNormMode_t in)
{
    if( in == HIPDNN_BATCHNORM_PER_ACTIVATION )
        return miopenBNPerActivation;
    else if( in == HIPDNN_BATCHNORM_SPATIAL )
        return miopenBNSpatial;
    
    return miopenBNPerActivation;
}

//=============================================================================

hipdnnStatus_t  hipTomiopenActivationMode(hipdnnActivationMode_t in, 
                                        miopenActivationMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    
    switch(in)
    {
    case HIPDNN_ACTIVATION_SIGMOID:
        *out = miopenActivationLOGISTIC;
        break;
    
    case HIPDNN_ACTIVATION_RELU:
        *out = miopenActivationRELU;
        break;
    
    case HIPDNN_ACTIVATION_TANH:
        *out = miopenActivationTANH;
        break;
        
    case HIPDNN_ACTIVATION_PATHTRU:
        *out =  miopenActivationPATHTRU;
        break;
        
    case HIPDNN_ACTIVATION_SOFTRELU:
        *out =  miopenActivationSOFTRELU;
        break;
        
    case HIPDNN_ACTIVATION_ABS:
        *out =  miopenActivationABS;
        break;
        
    case HIPDNN_ACTIVATION_POWER:
        *out = miopenActivationPOWER;
        break;
        
    case HIPDNN_ACTIVATION_ELU:
	retVal = HIPDNN_STATUS_NOT_SUPPORTED;
	break;

    case HIPDNN_ACTIVATION_CLIPPED_RELU:
	retVal = HIPDNN_STATUS_NOT_SUPPORTED;
	break;

    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}


//=============================================================================

hipdnnStatus_t  hipTomiopenConvolutionFwdAlgo(  hipdnnConvolutionFwdAlgo_t in, 
                                                miopenConvFwdAlgorithm_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    
    case HIPDNN_CONVOLUTION_FWD_ALGO_GEMM:
        *out = miopenConvolutionFwdAlgoGEMM;
        break;  
    case HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT:
        *out = miopenConvolutionFwdAlgoDirect;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_FFT:
        *out = miopenConvolutionFwdAlgoFFT;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
        *out = miopenConvolutionFwdAlgoWinograd;
        break;
    /*case HIPDNN_CONVOLUTION_FWD_ALGO_COUNT:
    case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
    case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
    case HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
    case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:*/ //TODO: will be added in future
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    
    return retVal;
} 

hipdnnStatus_t  miopenTohipConvolutionFwdAlgo(  miopenConvFwdAlgorithm_t  in, 
                                                hipdnnConvolutionFwdAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    
    case miopenConvolutionFwdAlgoGEMM :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_GEMM;
        break;  
    case miopenConvolutionFwdAlgoDirect :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT;
        break;
    case miopenConvolutionFwdAlgoFFT :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_FFT;
        break;
    case miopenConvolutionFwdAlgoWinograd :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    
    return retVal;
} 


int ConvolutionFwdAlgoCount()
{
    return 4;
}

// call ConvolutionFwdAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionFwdAlgo_t  GetConvolutionFwdAlgo(int i)
{
    hipdnnConvolutionFwdAlgo_t retVal;
    miopenConvFwdAlgorithm_t mialgo;
    
    if( i < ConvolutionFwdAlgoCount() )
    {
        mialgo = (miopenConvFwdAlgorithm_t)i;
    }
    else
    {
        //for protection
        mialgo = (miopenConvFwdAlgorithm_t)miopenConvolutionFwdAlgoWinograd;
    }
    miopenTohipConvolutionFwdAlgo(mialgo,&retVal);
    
    return retVal;
}

hipdnnConvolutionFwdAlgo_t GetConvolutionFwdAlgo(hipdnnConvolutionFwdAlgoPerf_t* pref)
{
    hipdnnConvolutionFwdAlgo_t algo;
    miopenTohipConvolutionFwdAlgo(((miopenConvAlgoPerf_t*)pref)->fwd_algo, &algo);
    return algo;
}

//=============================================================================

hipdnnStatus_t  hipTomiopenConvolutionBwdFilterAlgo(    hipdnnConvolutionBwdFilterAlgo_t in, 
                                                        miopenConvBwdWeightsAlgorithm_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
        *out = miopenConvolutionBwdWeightsAlgoGEMM;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
        *out = miopenConvolutionBwdWeightsAlgoDirect;
        break;
    /*case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:      
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:*/ //TODO: will be added in future
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    
    return retVal;
} 

hipdnnStatus_t  miopenTohipConvolutionBwdFilterAlgo(    miopenConvBwdWeightsAlgorithm_t  in, 
                                                        hipdnnConvolutionBwdFilterAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case miopenConvolutionBwdWeightsAlgoGEMM :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        break;
    case miopenConvolutionBwdWeightsAlgoDirect :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        break;
    }
    
    return retVal;
} 

                                                
int ConvolutionBwdFilterAlgoCount()
{
    return (int)2;
}
                                                        
// call ConvolutionBwdFilterAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdFilterAlgo_t  GetConvolutionBwdFilterAlgo(int i)
{
    hipdnnConvolutionBwdFilterAlgo_t retVal;
    miopenConvBwdWeightsAlgorithm_t mialgo;
    
    if( i < ConvolutionBwdFilterAlgoCount() )
    {
        mialgo = (miopenConvBwdWeightsAlgorithm_t)i;
    }
    else
    {
        //for protection
        mialgo = (miopenConvBwdWeightsAlgorithm_t)miopenConvolutionBwdWeightsAlgoGEMM;
    }
    miopenTohipConvolutionBwdFilterAlgo(mialgo,&retVal);
    
    return retVal;
}



//=============================================================================


hipdnnStatus_t  hipTomiopenConvolutionBwdDataAlgo(  hipdnnConvolutionBwdDataAlgo_t in, 
                                                    miopenConvBwdDataAlgorithm_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {   
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0:
        *out = miopenConvolutionBwdDataAlgoDirect;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1:
        *out = miopenConvolutionBwdDataAlgoWinograd;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
        *out = miopenConvolutionBwdDataAlgoWinograd;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
        *out = miopenConvolutionBwdDataAlgoWinograd;
        break;    
    /*case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:*/ //TODO: to be added in future
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    return retVal;
} 

hipdnnStatus_t  miopenTohipConvolutionBwdDataAlgo(  miopenConvBwdDataAlgorithm_t in, 
                                                    hipdnnConvolutionBwdDataAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {   
    case miopenConvolutionBwdDataAlgoDirect :
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        break;
    case miopenConvolutionBwdDataAlgoWinograd :
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
        break;
    }
    return retVal;
} 


int ConvolutionBwdDataAlgoCount()
{
    return (int)2;
}

// call ConvolutionBwdDataAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdDataAlgo_t  GetConvolutionBwdDataAlgo(int i)
{
    hipdnnConvolutionBwdDataAlgo_t retVal;
    miopenConvBwdDataAlgorithm_t mialgo;
    
    if( i < ConvolutionBwdDataAlgoCount() )
    {
        mialgo = (miopenConvBwdDataAlgorithm_t)i;
    }
    else
    {
        //for protection
        mialgo = (miopenConvBwdDataAlgorithm_t)miopenConvolutionBwdDataAlgoWinograd;
    }
    miopenTohipConvolutionBwdDataAlgo(mialgo,&retVal);
    
    return retVal;
}

//=============================================================================

hipdnnStatus_t  hipSoftmaxModeSupported( hipdnnSoftmaxMode_t in)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
        //HGSOS
    case HIPDNN_SOFTMAX_MODE_INSTANCE:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_SOFTMAX_MODE_CHANNEL:
        break;
    }
    return retVal;
} 

hipdnnStatus_t  SoftmaxAlgorithmSupported(  hipdnnSoftmaxAlgorithm_t in)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_SOFTMAX_FAST:
    case HIPDNN_SOFTMAX_ACCURATE:
        break;
    case HIPDNN_SOFTMAX_LOG:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    return retVal;
} 


// miopen does not define tensor format, 
// implicitly HIPDNN_TENSOR_NCHW only
hipdnnStatus_t  hipTensorFormatSupported(   hipdnnTensorFormat_t in)
{
    if( in == HIPDNN_TENSOR_NCHW )
        return HIPDNN_STATUS_SUCCESS;
    else
        return HIPDNN_STATUS_NOT_SUPPORTED;
} 


hipdnnStatus_t  ConvolutionFwdPreferenceSupported(  hipdnnConvolutionFwdPreference_t in )
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST:
        break;
    case HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    return retVal;
} 


hipdnnStatus_t  ConvolutionBwdFilterPreferenceSupported(    hipdnnConvolutionBwdFilterPreference_t in)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST:
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    return retVal;
} 



//=============================================================================

HIPDNN_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t *handle)
{
    sConvolutionForwardAlgorithmWorkspace = 0;
    
    return miopenTohipdnnStatus(miopenCreate(reinterpret_cast<miopenHandle_t *>(handle)));
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    if( sConvolutionForwardAlgorithmWorkspace != 0 )
    {
        hipFree(sConvolutionForwardAlgorithmWorkspace);
        sConvolutionForwardAlgorithmWorkspace = 0;
    }
    
    return miopenTohipdnnStatus(miopenDestroy((miopenHandle_t)handle));
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipdnnStream_t streamId)
{
    return miopenTohipdnnStatus(miopenSetStream((miopenHandle_t)handle, streamId));
}

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipdnnStream_t *streamId)
{
    return miopenTohipdnnStatus(miopenGetStream((miopenHandle_t)handle, streamId));
}

size_t hipdnnGetVersion()
{
    return 6000;
}

hipdnnStatus_t hipdnnCreateTensorDescriptor(hipdnnTensorDescriptor_t *tensorDesc)
{
    return miopenTohipdnnStatus(miopenCreateTensorDescriptor(reinterpret_cast<miopenTensorDescriptor_t*>(tensorDesc)));
}

//=============================================================================

hipdnnStatus_t  hipdnnSetTensor4dDescriptor(    hipdnnTensorDescriptor_t tensorDesc,
                                                hipdnnTensorFormat_t format, 
                                                hipdnnDataType_t dataType,
                                                int n, int c, int h, int w) 
{

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t miDT;
    
    retVal = hipTensorFormatSupported(format);
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    retVal = hipTomiopenDataType(dataType, &miDT);
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    return miopenTohipdnnStatus(
                miopenSet4dTensorDescriptor((miopenTensorDescriptor_t)tensorDesc,
                                            miDT,
                                            n, c, h, w));
                    
}   
            
//=============================================================================

hipdnnStatus_t hipdnnGetTensor4dDescriptor( hipdnnTensorDescriptor_t tensorDesc,
                                            hipdnnDataType_t *dataType,
                                            int *n, int *c, int *h, int *w,
                                            int *nStride, int *cStride,
                                            int *hStride, int *wStride)
{
    miopenDataType_t midT;  
    hipdnnStatus_t  retVal;                             
                                            
    retVal = miopenTohipdnnStatus(
                miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)tensorDesc,
                                            &midT,
                                            n, c, h, w,
                                            nStride, cStride,
                                            hStride, wStride));

    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    return miopenTohipDataType(midT, dataType);
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyTensorDescriptor(hipdnnTensorDescriptor_t tensorDesc)
{
    return miopenTohipdnnStatus(miopenDestroyTensorDescriptor((miopenTensorDescriptor_t)tensorDesc));
}

//=============================================================================

//dstValue = alpha[0]*srcValue + beta[0]*priorDstValue

hipdnnStatus_t hipdnnAddTensor(hipdnnHandle_t handle,
                                const void *alpha,
                                const hipdnnTensorDescriptor_t
                                aDesc, const void *A,
                                const void *beta,
                                const hipdnnTensorDescriptor_t
                                cDesc, void *C)
{
    miopenTensorOp_t tensorOp = miopenTensorOpAdd;
    int alpha2 = 0;

    return miopenTohipdnnStatus(miopenOpTensor((miopenHandle_t)handle,
                                    tensorOp,
                                    alpha,
                                    (miopenTensorDescriptor_t)aDesc, A,
                                    beta,
                                    (miopenTensorDescriptor_t)cDesc, C,
                                    &alpha2,
                                    (miopenTensorDescriptor_t)cDesc, C));
}

//=============================================================================

hipdnnStatus_t
hipdnnOpTensor( hipdnnHandle_t handle,
                const hipdnnOpTensorDescriptor_t opTensorDesc,
                const void *alpha1,
                const hipdnnTensorDescriptor_t aDesc, 
                const void *A,
                const void *alpha2,
                const hipdnnTensorDescriptor_t bDesc, 
                const void *B,
                const void *beta,
                const hipdnnTensorDescriptor_t
                cDesc, void *C)
{
    return miopenTohipdnnStatus(miopenOpTensor( (miopenHandle_t)handle,*(reinterpret_cast<miopenTensorOp_t *>(const_cast<hipdnnOpTensorDescriptor_t *>(&opTensorDesc))), alpha1, 
                                                (miopenTensorDescriptor_t)aDesc,A,alpha2,(miopenTensorDescriptor_t)bDesc, B,beta, (miopenTensorDescriptor_t)cDesc, C));
}
//=============================================================================

hipdnnStatus_t hipdnnSetTensor( hipdnnHandle_t handle,
                                const hipdnnTensorDescriptor_t yDesc, 
                                void *y,
                                const void *valuePtr)
{
    return miopenTohipdnnStatus(miopenSetTensor((miopenHandle_t) handle, (miopenTensorDescriptor_t)yDesc, y,valuePtr));
}

//=============================================================================

hipdnnStatus_t
hipdnnScaleTensor(  hipdnnHandle_t handle,
                    const hipdnnTensorDescriptor_t yDesc, 
                    void *y,
                    const void *alpha)
{
    return miopenTohipdnnStatus( miopenScaleTensor((miopenHandle_t)handle,(miopenTensorDescriptor_t)yDesc, y,alpha));
}

//=============================================================================

hipdnnStatus_t
hipdnnCreateFilterDescriptor(hipdnnFilterDescriptor_t *filterDesc)
{
    //in miopen a filter descriptor is just a typedef to a tensor descriptor
    return hipdnnCreateTensorDescriptor(reinterpret_cast<hipdnnTensorDescriptor_t*>(filterDesc));
}

//=============================================================================

hipdnnStatus_t
hipdnnCreateConvolutionDescriptor(hipdnnConvolutionDescriptor_t *convDesc)
{
    return miopenTohipdnnStatus( miopenCreateConvolutionDescriptor(reinterpret_cast<miopenConvolutionDescriptor_t*>(convDesc)));
}
//=============================================================================

hipdnnStatus_t
hipdnnSetConvolution2dDescriptor(hipdnnConvolutionDescriptor_t convDesc,
                                int pad_h, int pad_w,
                                int u, int v,
                                int upscalex, int upscaley,
                                hipdnnConvolutionMode_t mode,
                                hipdnnDataType_t computeType)  
{
    
    return miopenTohipdnnStatus(
            miopenInitConvolutionDescriptor((miopenConvolutionDescriptor_t)convDesc,
                                            hipTomiopenConvolutionMode(mode), 
                                            pad_h, pad_w,
                                            u, v,
                                            upscalex, upscaley));
}


//=============================================================================

hipdnnStatus_t
hipdnnGetConvolution2dDescriptor(const hipdnnConvolutionDescriptor_t convDesc,
                                int *pad_h, int *pad_y,
                                int *u, int *v,
                                int *upscalex, int *upscaley,
                                hipdnnConvolutionMode_t *mode,
                                hipdnnDataType_t *computeType)
{
    
    hipdnnStatus_t  retVal;       
    miopenConvolutionMode_t miMode;

    retVal = miopenTohipdnnStatus( miopenGetConvolutionDescriptor((miopenConvolutionDescriptor_t)convDesc,
                                                                  &miMode,  
                                                                  pad_h, pad_y,
                                                                  u, v,
                                                                  upscalex, upscaley));
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    *mode = miopenTohipConvolutionMode(miMode);
    
    //HGSOS miopen does not support this. Any better way to do this? 
    //which call should define the type?
    *computeType = HIPDNN_DATA_FLOAT;
    
    return retVal;
}



//===========

hipdnnStatus_t
hipdnnGetConvolution2dForwardOutputDim(
                const hipdnnConvolutionDescriptor_t convDesc,
                const hipdnnTensorDescriptor_t inputTensorDesc,
                const hipdnnFilterDescriptor_t filterDesc,
                int *n, int *c, int *h, int *w)
{
return  miopenTohipdnnStatus(
            miopenGetConvolutionForwardOutputDim(
                            (miopenConvolutionDescriptor_t)convDesc,  //HGSOSOS should be const in miopen.
                            (miopenTensorDescriptor_t)inputTensorDesc,
                            (miopenTensorDescriptor_t)filterDesc,
                            n, c, h, w));
}

//==============================================================================

hipdnnStatus_t
hipdnnDestroyConvolutionDescriptor(hipdnnConvolutionDescriptor_t convDesc)
{
return  miopenTohipdnnStatus(
            miopenDestroyConvolutionDescriptor((miopenConvolutionDescriptor_t)convDesc));
}

//===============================================================================


hipdnnStatus_t
hipdnnFindConvolutionForwardAlgorithm(  hipdnnHandle_t handle,
                                        const hipdnnTensorDescriptor_t xDesc,
                                        const hipdnnFilterDescriptor_t wDesc,
                                        const hipdnnConvolutionDescriptor_t convDesc,
                                        const hipdnnTensorDescriptor_t yDesc,
                                        const int requestedAlgoCount,
                                        int *returnedAlgoCount,
                                        hipdnnConvolutionFwdAlgoPerf_t *perfResults)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
#ifdef NOTYET   


    miopenConvFwdAlgorithm_t mialgo;
    hipdnnStatus_t retVal;
   
    //in miopen, workspace size does not depend on algo.
    retVal =  miopenTohipdnnStatus(
                miopenConvolutionForwardGetWorkSpaceSize(  (miopenHandle_t)handle,
                                                            (miopenTensorDescriptor_t)wDesc,  
                                                            (miopenTensorDescriptor_t)xDesc,  
                                                            (miopenConvolutionDescriptor_t)convDesc,
                                                            (miopenTensorDescriptor_t)yDesc,
                                                            sizeInBytes));
                                                            
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    
    hipMalloc((void**)&sConvolutionForwardAlgorithmWorkspace, sizeInBytes);
    
    //HGSOS //NOTYET dont know how to get x,y,w from the descriptors but it should be possible.
    
    return  hipdnnFindConvolutionForwardAlgorithmEx((miopenHandle_t)handle,
                                                xDesc,
                                                const void *x,
                                                wDesc,
                                                const void *w,
                                                (miopenConvolutionDescriptor_t)convDesc,
                                                yDesc,
                                                void *y,
                                                requestedAlgoCount,
                                                returnedAlgoCount,
                                                reinterpret_cast<miopenConvAlgoPerf_t *>(perfResults),
                                                void *workSpace,
                                                size_t workSpaceSizeInBytes)
#endif                                              
}

hipdnnStatus_t
hipdnnGetConvolutionForwardAlgorithm(hipdnnHandle_t handle,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const hipdnnFilterDescriptor_t wDesc,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    hipdnnConvolutionFwdPreference_t preference,
                                    size_t memoryLimitInBytes,
                                    hipdnnConvolutionFwdAlgo_t *algo)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
#ifdef NOTYET   

    miopenConvFwdAlgorithm_t mialgo;
    hipdnnStatus_t retVal;
   
    //in miopen, workspace size does not depend on algo.
    retVal =  miopenTohipdnnStatus(
                miopenConvolutionForwardGetWorkSpaceSize(  (miopenHandle_t)handle,
                                                            (miopenTensorDescriptor_t)wDesc,  
                                                           (miopenTensorDescriptor_t) xDesc,  
                                                            (miopenConvolutionDescriptor_t)convDesc,
                                                           (miopenTensorDescriptor_t) yDesc,
                                                            sizeInBytes));
                                                            
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    
    hipMalloc((void**)&sConvolutionForwardAlgorithmWorkspace, sizeInBytes);
    
//HGSOS //NOTYET dont know how to get x,y,w from the descriptors but it should be possible.

    return  hipdnnFindConvolutionForwardAlgorithmEx((miopenHandle_t)handle,
                                                xDesc,
                                                const void *x,
                                                wDesc,
                                                const void *w,
                                                (miopenConvolutionDescriptor_t)convDesc,
                                                yDesc,
                                                void *y,
                                                requestedAlgoCount,
                                                returnedAlgoCount,
                                                perfResults,
                                                void *workSpace,
                                                size_t workSpaceSizeInBytes)
#endif  
}

hipdnnStatus_t
hipdnnFindConvolutionForwardAlgorithmEx(hipdnnHandle_t handle,
                                        const hipdnnTensorDescriptor_t xDesc,
                                        const void *x,
                                        const hipdnnFilterDescriptor_t wDesc,
                                        const void *w,
                                        const hipdnnConvolutionDescriptor_t convDesc,
                                        const hipdnnTensorDescriptor_t yDesc,
                                        void *y,
                                        const int requestedAlgoCount,
                                        int *returnedAlgoCount,
                                        hipdnnConvolutionFwdAlgoPerf_t *perfResults,
                                        void *workSpace,
                                        size_t workSpaceSizeInBytes)
{
    size_t size;
    hipdnnStatus_t retVal;
    retVal =  miopenTohipdnnStatus(
                   miopenConvolutionForwardGetWorkSpaceSize((miopenHandle_t)handle,
                                                            (miopenTensorDescriptor_t)wDesc,  
                                                            (miopenTensorDescriptor_t)xDesc,  
                                                            (miopenConvolutionDescriptor_t)convDesc,
                                                            (miopenTensorDescriptor_t)yDesc,
                                                            &size));
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    
    hipMalloc((void**)&sConvolutionForwardAlgorithmWorkspace, size);
    return  miopenTohipdnnStatus(
                miopenFindConvolutionForwardAlgorithm((miopenHandle_t)handle,
                                                    (miopenTensorDescriptor_t)xDesc,
                                                    x,
                                                    (miopenTensorDescriptor_t)wDesc,
                                                    w,
                                                    (miopenConvolutionDescriptor_t)convDesc,
                                                    (miopenTensorDescriptor_t)yDesc,
                                                    y,
                                                    requestedAlgoCount,
                                                    returnedAlgoCount,
                                                    reinterpret_cast<miopenConvAlgoPerf_t *>(perfResults),
                                                    sConvolutionForwardAlgorithmWorkspace,
                                                    size,
                                                    true //exhaustiveSearch
                                                    ));
}


//=========================================!

hipdnnStatus_t
hipdnnGetConvolutionForwardWorkspaceSize(   hipdnnHandle_t handle,
                                            const hipdnnTensorDescriptor_t xDesc,  
                                            const hipdnnFilterDescriptor_t wDesc,  
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnTensorDescriptor_t yDesc,
                                            hipdnnConvolutionFwdAlgo_t algo,  
                                            size_t *sizeInBytes)
{
    miopenConvFwdAlgorithm_t mialgo;
    hipdnnStatus_t retVal = hipTomiopenConvolutionFwdAlgo(algo, &mialgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    //in miopen, workspace size does not depend on algo.
    return  miopenTohipdnnStatus(
                miopenConvolutionForwardGetWorkSpaceSize(  (miopenHandle_t)handle,
                                                            (miopenTensorDescriptor_t)wDesc,  
                                                            (miopenTensorDescriptor_t)xDesc,  
                                                            (miopenConvolutionDescriptor_t)convDesc,
                                                            (miopenTensorDescriptor_t)yDesc,
                                                            sizeInBytes));

}


//=============================================================================

hipdnnStatus_t
hipdnnConvolutionForward(   hipdnnHandle_t handle,
                            const void *alpha,
                            const hipdnnTensorDescriptor_t xDesc,
                            const void *x,
                            const hipdnnFilterDescriptor_t wDesc,
                            const void *w,
                            const hipdnnConvolutionDescriptor_t convDesc,
                            hipdnnConvolutionFwdAlgo_t algo,
                            void *workSpace,
                            size_t workSpaceSizeInBytes,  
                            const void *beta,
                            const hipdnnTensorDescriptor_t yDesc,
                            void *y)
{
    miopenConvFwdAlgorithm_t mialgo;
    hipdnnStatus_t retVal = hipTomiopenConvolutionFwdAlgo(algo, &mialgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return  miopenTohipdnnStatus(
                    miopenConvolutionForward((miopenHandle_t)handle,
                                            alpha,
                                            (miopenTensorDescriptor_t)xDesc,
                                            x,
                                            (miopenTensorDescriptor_t)wDesc,
                                            w,
                                            (miopenConvolutionDescriptor_t)convDesc,
                                            mialgo,
                                            beta,
                                            (miopenTensorDescriptor_t)yDesc,
                                            y,
                                            workSpace,
                                            workSpaceSizeInBytes));  

}

//=============================================================================


hipdnnStatus_t
hipdnnConvolutionBackwardBias(  hipdnnHandle_t handle,
                                const void *alpha,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const void *beta,
                                const hipdnnTensorDescriptor_t dbDesc,
                                void *db)
{
    return  miopenTohipdnnStatus(
                miopenConvolutionBackwardBias( (miopenHandle_t)handle,
                                                alpha,
                                                (miopenTensorDescriptor_t)dyDesc,
                                                dy,
                                                beta,
                                                (miopenTensorDescriptor_t)dbDesc,
                                                db));
}

//================HGSOS======================
hipdnnStatus_t
hipdnnFindConvolutionBackwardFilterAlgorithm(   hipdnnHandle_t handle,
                                                const hipdnnTensorDescriptor_t xDesc,
                                                const hipdnnTensorDescriptor_t dyDesc,
                                                const hipdnnConvolutionDescriptor_t convDesc,
                                                const hipdnnFilterDescriptor_t dwDesc,
                                                const int requestedAlgoCount,
                                                int *returnedAlgoCount,
                                                hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    
    return HIPDNN_STATUS_NOT_SUPPORTED;
    
#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif

}

hipdnnStatus_t
hipdnnGetConvolutionBackwardFilterAlgorithm(hipdnnHandle_t handle,
                                            const hipdnnTensorDescriptor_t xDesc,
                                            const hipdnnTensorDescriptor_t dyDesc,
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnFilterDescriptor_t dwDesc,
                                            hipdnnConvolutionBwdFilterPreference_t preference,
                                            size_t memoryLimitInBytes,
                                            hipdnnConvolutionBwdFilterAlgo_t *algo)
{
    
#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}


hipdnnStatus_t
hipdnnFindConvolutionBackwardFilterAlgorithmEx( hipdnnHandle_t handle,
                                                const hipdnnTensorDescriptor_t xDesc,
                                                const void *x,
                                                const hipdnnTensorDescriptor_t dyDesc,
                                                const void *dy,
                                                const hipdnnConvolutionDescriptor_t convDesc,
                                                const hipdnnFilterDescriptor_t dwDesc,
                                                void *dw,
                                                const int requestedAlgoCount,
                                                int *returnedAlgoCount,
                                                hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                                                void *workSpace,
                                                size_t workSpaceSizeInBytes)
{
                                            
    return  miopenTohipdnnStatus(
                miopenFindConvolutionBackwardWeightsAlgorithm(  (miopenHandle_t)handle,
                                                                (miopenTensorDescriptor_t)dyDesc,
                                                                dy,
                                                                (miopenTensorDescriptor_t)xDesc,
                                                                x,
                                                                (miopenConvolutionDescriptor_t)convDesc,
                                                                (miopenTensorDescriptor_t)dwDesc,
                                                                dw,
                                                                requestedAlgoCount,
                                                                returnedAlgoCount,
                                                                reinterpret_cast<miopenConvAlgoPerf_t *>(perfResults),
                                                                workSpace,
                                                                workSpaceSizeInBytes,
                                                                true //exhaustiveSearch
                                                                )); 

}

//=================HGSOS======================!

hipdnnStatus_t
hipdnnGetConvolutionBackwardFilterWorkspaceSize(    hipdnnHandle_t handle,
                                                    const hipdnnTensorDescriptor_t xDesc,
                                                    const hipdnnTensorDescriptor_t dyDesc,
                                                    const hipdnnConvolutionDescriptor_t convDesc,
                                                    const hipdnnFilterDescriptor_t dwDesc,  
                                                    hipdnnConvolutionBwdFilterAlgo_t algo,
                                                    size_t *sizeInBytes)
{

// in miopen, workspace size doesnt depend on algo.

    return  miopenTohipdnnStatus(
                miopenConvolutionBackwardWeightsGetWorkSpaceSize( (miopenHandle_t)handle,
                (miopenTensorDescriptor_t)dyDesc,
                (miopenTensorDescriptor_t)xDesc,
                (miopenConvolutionDescriptor_t)convDesc,
                (miopenTensorDescriptor_t)dwDesc,
                sizeInBytes));

}

//=============================================================================

hipdnnStatus_t
hipdnnConvolutionBackwardFilter(    hipdnnHandle_t handle,
                                    const void *alpha,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x,
                                    const hipdnnTensorDescriptor_t dyDesc,
                                    const void *dy,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    hipdnnConvolutionBwdFilterAlgo_t algo,
                                    void *workSpace,
                                    size_t workSpaceSizeInBytes,
                                    const void *beta,
                                    const hipdnnFilterDescriptor_t dwDesc,
                                    void *dw)
{
                                            
    miopenConvBwdWeightsAlgorithm_t mialgo;
    hipdnnStatus_t retVal = hipTomiopenConvolutionBwdFilterAlgo(algo, &mialgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return  miopenTohipdnnStatus(
                miopenConvolutionBackwardWeights(  (miopenHandle_t)handle,
                                                    alpha,
                                                    (miopenTensorDescriptor_t)dyDesc,
                                                    dy,
                                                    (miopenTensorDescriptor_t)xDesc,
                                                    x,
                                                    (miopenConvolutionDescriptor_t)convDesc,
                                                    mialgo,
                                                    beta,
                                                    (miopenTensorDescriptor_t)dwDesc,
                                                    dw,
                                                    workSpace,
                                                    workSpaceSizeInBytes));
}

//=============================================================================


hipdnnStatus_t
hipdnnGetConvolutionBackwardDataWorkspaceSize(  hipdnnHandle_t handle,
                                                const hipdnnFilterDescriptor_t wDesc,
                                                const hipdnnTensorDescriptor_t dyDesc,
                                                const hipdnnConvolutionDescriptor_t convDesc,
                                                const hipdnnTensorDescriptor_t dxDesc,
                                                hipdnnConvolutionBwdDataAlgo_t algo,  
                                                size_t *sizeInBytes)
{
    //does not depend on algo in miopen
    
    return  miopenTohipdnnStatus(
        miopenConvolutionBackwardDataGetWorkSpaceSize( (miopenHandle_t)handle,
                                                        (miopenTensorDescriptor_t)dyDesc,
                                                        (miopenTensorDescriptor_t)wDesc,
                                                        (miopenConvolutionDescriptor_t)convDesc,
                                                        (miopenTensorDescriptor_t)dxDesc,
                                                        sizeInBytes));
}

//=============================================================================

hipdnnStatus_t
hipdnnFindConvolutionBackwardDataAlgorithm( hipdnnHandle_t handle,
                                            const hipdnnFilterDescriptor_t wDesc,
                                            const hipdnnTensorDescriptor_t dyDesc,
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnTensorDescriptor_t dxDesc,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            hipdnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    return  HIPDNN_STATUS_NOT_SUPPORTED;
    
#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif

}


hipdnnStatus_t
hipdnnGetConvolutionBackwardDataAlgorithm(  hipdnnHandle_t handle,
                                            const hipdnnFilterDescriptor_t wDesc,
                                            const hipdnnTensorDescriptor_t dyDesc,
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnTensorDescriptor_t dxDesc,
                                            hipdnnConvolutionBwdDataPreference_t preference,
                                            size_t memoryLimitInBytes,
                                            hipdnnConvolutionBwdDataAlgo_t *algo)
{
    return  HIPDNN_STATUS_NOT_SUPPORTED;
    
#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif
}

hipdnnStatus_t
hipdnnFindConvolutionBackwardDataAlgorithmEx(hipdnnHandle_t handle,
                                            const hipdnnFilterDescriptor_t wDesc,
                                            const void *w,
                                            const hipdnnTensorDescriptor_t dyDesc,
                                            const void *dy,
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnTensorDescriptor_t dxDesc,
                                            void *dx,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            hipdnnConvolutionBwdDataAlgoPerf_t *perfResults,
                                            void *workSpace,
                                            size_t workSpaceSizeInBytes)

{

    return  miopenTohipdnnStatus(
                miopenFindConvolutionBackwardDataAlgorithm((miopenHandle_t)handle,
                (miopenTensorDescriptor_t)dyDesc,
                dy,
                (miopenTensorDescriptor_t)wDesc,
                w,
                (miopenConvolutionDescriptor_t)convDesc,
                (miopenTensorDescriptor_t)dxDesc,
                dx,
                requestedAlgoCount,
                returnedAlgoCount,
                reinterpret_cast<miopenConvAlgoPerf_t*>(perfResults),
                workSpace,
                workSpaceSizeInBytes,
                true // exhaustiveSearch
                ));
}

//=============================================================================


hipdnnStatus_t
hipdnnConvolutionBackwardData(  hipdnnHandle_t handle,
                                const void *alpha,
                                const hipdnnFilterDescriptor_t wDesc,
                                const void *w,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnConvolutionDescriptor_t convDesc,
                                hipdnnConvolutionBwdDataAlgo_t algo,
                                void *workSpace,
                                size_t workSpaceSizeInBytes,
                                const void *beta,
                                const hipdnnTensorDescriptor_t dxDesc,
                                void *dx)
{
    miopenConvBwdDataAlgorithm_t mialgo;  
    hipdnnStatus_t retVal = hipTomiopenConvolutionBwdDataAlgo(algo, &mialgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return  miopenTohipdnnStatus(
                    miopenConvolutionBackwardData( (miopenHandle_t)handle,
                                                    alpha,
                                                    (miopenTensorDescriptor_t)dyDesc,
                                                    dy,
                                                    (miopenTensorDescriptor_t)wDesc,
                                                    w,
                                                    (miopenConvolutionDescriptor_t)convDesc,
                                                    mialgo,
                                                    beta,
                                                    (miopenTensorDescriptor_t)dxDesc,
                                                    dx,
                                                    workSpace,
                                                    workSpaceSizeInBytes));
}

//=============================================================================

hipdnnStatus_t
hipdnnSoftmaxForward(hipdnnHandle_t handle,
                    hipdnnSoftmaxAlgorithm_t algo,  
                    hipdnnSoftmaxMode_t mode,   
                    const void *alpha,
                    const hipdnnTensorDescriptor_t xDesc,
                    const void *x, 
                    const void *beta,
                    const hipdnnTensorDescriptor_t yDesc,
                    void *y)
{

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    retVal = SoftmaxAlgorithmSupported(algo);
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    retVal = hipSoftmaxModeSupported( mode );
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return  miopenTohipdnnStatus(
                miopenSoftmaxForward(  (miopenHandle_t)handle,
                                        alpha,
                                        (miopenTensorDescriptor_t)xDesc,
                                        x, beta,
                                        (miopenTensorDescriptor_t)yDesc,
                                        y));
}

//=============================================================================

hipdnnStatus_t
hipdnnSoftmaxBackward(  hipdnnHandle_t handle,
                        hipdnnSoftmaxAlgorithm_t algo,  
                        hipdnnSoftmaxMode_t mode,
                        const void *alpha,
                        const hipdnnTensorDescriptor_t yDesc,
                        const void *y,
                        const hipdnnTensorDescriptor_t dyDesc,
                        const void *dy, const void *beta,
                        const hipdnnTensorDescriptor_t dxDesc,
                        void *dx)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    retVal = SoftmaxAlgorithmSupported(algo);
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    retVal = hipSoftmaxModeSupported( mode );
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return  miopenTohipdnnStatus(
                miopenSoftmaxBackward( (miopenHandle_t)handle,
                                        alpha,
                                        (miopenTensorDescriptor_t)yDesc,
                                        y,
                                        (miopenTensorDescriptor_t)dyDesc,
                                        dy, 
                                        beta,
                                        (miopenTensorDescriptor_t)dxDesc,
                                        dx));
}

//=============================================================================


hipdnnStatus_t
hipdnnCreatePoolingDescriptor(hipdnnPoolingDescriptor_t *poolingDesc)
{
    return  miopenTohipdnnStatus(
                miopenCreatePoolingDescriptor(reinterpret_cast<miopenPoolingDescriptor_t *>(poolingDesc)));
}
//=============================================================================

hipdnnStatus_t
hipdnnSetPooling2dDescriptor(   hipdnnPoolingDescriptor_t poolingDesc,
                                hipdnnPoolingMode_t mode,
                                hipdnnNanPropagation_t maxpoolingNanOpt, 
                                int windowHeight,
                                int windowWidth,
                                int verticalPadding,
                                int horizontalPadding,
                                int verticalStride,
                                int horizontalStride)
{

    miopenPoolingMode_t miPMode;
    hipdnnStatus_t retVal = hipTomiopenPoolingMode(mode, &miPMode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    //HGSOS verify
    
    return miopenTohipdnnStatus(
                miopenSet2dPoolingDescriptor(  (miopenPoolingDescriptor_t) poolingDesc,
                                                miPMode,
                                                windowHeight,
                                                windowWidth,
                                                horizontalPadding, verticalPadding,
                                                horizontalStride, verticalStride));
    
}


//=============================================================================

hipdnnStatus_t
hipdnnGetPooling2dDescriptor(   const hipdnnPoolingDescriptor_t poolingDesc,
                                hipdnnPoolingMode_t *mode,
                                hipdnnNanPropagation_t *maxpoolingNanOpt, 
                                int *windowHeight,
                                int *windowWidth,
                                int *verticalPadding,
                                int *horizontalPadding,
                                int *verticalStride,
                                int *horizontalStride)
{
    hipdnnStatus_t retVal;
    miopenPoolingMode_t mipmmode;

    retVal = miopenTohipdnnStatus(  
                miopenGet2dPoolingDescriptor((miopenPoolingDescriptor_t)  poolingDesc,
                                                &mipmmode,
                                                windowHeight,
                                                windowWidth,
                                                horizontalPadding, horizontalPadding,
                                                horizontalStride, verticalStride));
                                            
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    *maxpoolingNanOpt = HIPDNN_PROPAGATE_NAN;
    
    return  miopenTohipPoolingMode( mipmmode, mode );
}


//=============================================================================

hipdnnStatus_t
hipdnnGetPooling2dForwardOutputDim( const hipdnnPoolingDescriptor_t poolingDesc,
                                    const hipdnnTensorDescriptor_t inputTensorDesc,
                                    int *n, int *c, int *h, int *w)
{
    return miopenTohipdnnStatus(
                miopenGetPoolingForwardOutputDim(  (miopenPoolingDescriptor_t)  poolingDesc,
                                                    (miopenTensorDescriptor_t)inputTensorDesc,
                                                    n, c, h, w));
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyPoolingDescriptor(hipdnnPoolingDescriptor_t poolingDesc)
{
    return miopenTohipdnnStatus(miopenDestroyPoolingDescriptor( (miopenPoolingDescriptor_t)poolingDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnPoolingForward(hipdnnHandle_t handle,
                                    const hipdnnPoolingDescriptor_t poolingDesc,
                                    const void *alpha,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x,
                                    const void *beta,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    void *y)
{
//    return HIPDNN_STATUS_NOT_SUPPORTED;
    int8_t* devptr = 0;
        
    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

    
    if( sDescToWorkspacePooling.find((miopenTensorDescriptor_t)yDesc) != sDescToWorkspacePooling.end() )
    {
        //HGSOS looks like the yDesc is used for the workspace, not the poolingDesc
        
        miStat = miopenPoolingGetWorkSpaceSize( (miopenTensorDescriptor_t)yDesc, &workSpaceSize);
        if(  miStat == miopenStatusSuccess )
        {
            hipMalloc( (void**)&devptr, workSpaceSize);
            sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc] = devptr;
            sDescToWorkspacePoolingSize[(miopenTensorDescriptor_t)yDesc] = workSpaceSize;
        }
        else
        {
            return miopenTohipdnnStatus(miStat);
        }
        
    }
    else
    {
        devptr = sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc];
        workSpaceSize = sDescToWorkspacePoolingSize[(miopenTensorDescriptor_t)yDesc];
    }

    return miopenTohipdnnStatus ( miopenPoolingForward(
                                       (miopenHandle_t)handle,
                                        (miopenPoolingDescriptor_t)poolingDesc,
                                        alpha,
                                        (miopenTensorDescriptor_t)xDesc,
                                        x, 
                                        beta,
                                        (miopenTensorDescriptor_t)yDesc,
                                        y, 
                                        false, // do_backward,
                                        (void *)devptr,  
                                        workSpaceSize)); 

}

//=================================!

hipdnnStatus_t
hipdnnPoolingBackward(  hipdnnHandle_t handle,
                        const hipdnnPoolingDescriptor_t poolingDesc,
                        const void *alpha,
                        const hipdnnTensorDescriptor_t yDesc,
                        const void *y,
                        const hipdnnTensorDescriptor_t dyDesc,
                        const void *dy,
                        const hipdnnTensorDescriptor_t xDesc,
                        const void *x, const void *beta,
                        const hipdnnTensorDescriptor_t dxDesc,
                        void *dx)
{
//    return HIPDNN_STATUS_NOT_SUPPORTED;
    
    int8_t* devptr = 0;
        
    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

    //HGSOS it appears that forward and backward pooling can reuse tha same map.
    
    if( sDescToWorkspacePooling.find((miopenTensorDescriptor_t)yDesc) != sDescToWorkspacePooling.end() )
    {
        //HGSOS looks like the yDesc is used for the workspace, not the poolingDesc
        
        miStat = miopenPoolingGetWorkSpaceSize((miopenTensorDescriptor_t) yDesc, &workSpaceSize);
        if(  miStat == miopenStatusSuccess )
        {
            hipMalloc( (void**)&devptr, workSpaceSize);
            sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc] = devptr;
            sDescToWorkspacePoolingSize[(miopenTensorDescriptor_t)yDesc] = workSpaceSize;
        }
        else
        {
            return miopenTohipdnnStatus(miStat);
        }
        
    }
    else
    {
        devptr = sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc];
        workSpaceSize = sDescToWorkspacePoolingSize[(miopenTensorDescriptor_t)yDesc];
    }

    return miopenTohipdnnStatus (
                miopenPoolingBackward(
                                        (miopenHandle_t)handle,
                                        (miopenPoolingDescriptor_t)poolingDesc,
                                        alpha,
                                        (miopenTensorDescriptor_t)yDesc,
                                        y,
                                        (miopenTensorDescriptor_t)dyDesc,
                                        dy,
                                        (miopenTensorDescriptor_t)xDesc,
                                        x, 
                                        beta,
                                        (miopenTensorDescriptor_t)dxDesc,
                                        dx, 
                                        devptr) ); //HGSOS  //NOTYET no worspace size!  const!!!????

}
//=============================================================================

hipdnnStatus_t
hipdnnCreateActivationDescriptor(hipdnnActivationDescriptor_t *activationDesc)
{
    return miopenTohipdnnStatus(miopenCreateActivationDescriptor(reinterpret_cast<miopenActivationDescriptor_t *>(activationDesc)));
}
//=============================================================================

hipdnnStatus_t
hipdnnSetActivationDescriptor(  hipdnnActivationDescriptor_t activationDesc, 
                                hipdnnActivationMode_t mode,
                                hipdnnNanPropagation_t reluNanOpt, 
                                double reluCeiling)
{
    miopenActivationMode_t mimode;
    hipTomiopenActivationMode(mode, &mimode);

    return miopenTohipdnnStatus (
                miopenSetActivationDescriptor(
                                              (miopenActivationDescriptor_t)activationDesc,
                                              mimode,
                                              0, //Alpha
                                              0, //Beta
                                              1)); //Power
}

//=============================================================================

//HGSOS may need another function to accommodate the 3 parameters.

hipdnnStatus_t
hipdnnGetActivationDescriptor(  const hipdnnActivationDescriptor_t activationDesc,
                                hipdnnActivationMode_t *mode,
                                hipdnnNanPropagation_t *reluNanOpt,  
                                double* reluCeiling)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
    
#ifdef NOTYET

//HGSOS //NOTYET fix the miopenSetActivationDescriptor api first

miopenStatus_t
miopenGetActivationDescriptor(
const miopenActivationDescriptor_t activDesc,
miopenActivationMode_t *mode,
double *activAlpha,
double *activBeta,
double *activPower)

    miopenActivationDescriptor_t miActDes;
    hipdnnStatus_t retVal;

    retVal = hipTomiopenActivationMode(activationDesc, &miActDes);

    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
#endif
    
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyActivationDescriptor(hipdnnActivationDescriptor_t activationDesc)
{
    return  miopenTohipdnnStatus(miopenDestroyActivationDescriptor((miopenActivationDescriptor_t)activationDesc));
}
//=================

hipdnnStatus_t
hipdnnActivationForward(    hipdnnHandle_t handle,
                            hipdnnActivationDescriptor_t activationDesc, //HGSOS not const in cudnn
                            const void *alpha,
                            const hipdnnTensorDescriptor_t xDesc,
                            const void *x,
                            const void *beta,
                            const hipdnnTensorDescriptor_t yDesc,
                            void *y)
{
    return  miopenTohipdnnStatus(
                        miopenActivationForward(
                                               (miopenHandle_t)handle,
                                                (miopenActivationDescriptor_t)activationDesc,
                                                alpha,
                                                (miopenTensorDescriptor_t)xDesc,
                                                x,
                                                beta,
                                                (miopenTensorDescriptor_t)yDesc,
                                                y));
}
//======================

hipdnnStatus_t
hipdnnActivationBackward(   hipdnnHandle_t handle,
                            hipdnnActivationDescriptor_t activationDesc,  //HGSOS const missing in cuda
                            const void *alpha,
                            const hipdnnTensorDescriptor_t yDesc,
                            const void *y,
                            const hipdnnTensorDescriptor_t dyDesc,
                            const void *dy,
                            const hipdnnTensorDescriptor_t xDesc,
                            const void *x,
                            const void *beta,
                            const hipdnnTensorDescriptor_t dxDesc,
                            void *dx)
{
    return  miopenTohipdnnStatus(
                miopenActivationBackward(  (miopenHandle_t)handle,
                                            (miopenActivationDescriptor_t)activationDesc,
                                            alpha,
                                            (miopenTensorDescriptor_t)yDesc,
                                            y,
                                            (miopenTensorDescriptor_t)dyDesc,
                                            dy,
                                            (miopenTensorDescriptor_t)xDesc,
                                            x, beta,
                                            (miopenTensorDescriptor_t)dxDesc,
                                            dx));
}
//=============================================================================

hipdnnStatus_t
hipdnnCreateLRNDescriptor(hipdnnLRNDescriptor_t *normDesc)
{
    return  miopenTohipdnnStatus(miopenCreateLRNDescriptor(reinterpret_cast<miopenLRNDescriptor_t *>(normDesc)));
}
//=============================================================================

hipdnnStatus_t
hipdnnSetLRNDescriptor( hipdnnLRNDescriptor_t normDesc,
                        hipdnnLRNMode_t mode,  
                        unsigned lrnN, double lrnAlpha,
                        double lrnBeta, double lrnK)
{
    
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;
    
    retVal = hipTomiopenLRNMode( mode, &mimode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return  miopenTohipdnnStatus(
                miopenSetLRNDescriptor( (miopenLRNDescriptor_t)normDesc,
                                        mimode,  
                                        lrnN, lrnAlpha,
                                        lrnBeta, lrnK));

}

//================

hipdnnStatus_t
hipdnnGetLRNDescriptor( hipdnnLRNDescriptor_t normDesc,
                        hipdnnLRNMode_t *mode, 
                        unsigned* lrnN, 
                        double* lrnAlpha,
                        double* lrnBeta, 
                        double* lrnK)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;
    
    retVal = miopenTohipdnnStatus(
                miopenGetLRNDescriptor( (miopenLRNDescriptor_t)  normDesc,  
                                        &mimode, 
                                        lrnN, 
                                        lrnAlpha,
                                        lrnBeta, 
                                        lrnK));
                                    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return miopenTohipLRNMode(mimode, mode);
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyLRNDescriptor(hipdnnLRNDescriptor_t normDesc)
{
    return miopenTohipdnnStatus(miopenDestroyLRNDescriptor((miopenLRNDescriptor_t)  normDesc));
}

//=============================================================================

hipdnnStatus_t
hipdnnLRNCrossChannelForward(   hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode,  
                                const void* alpha,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x,
                                const void *beta,
                                const hipdnnTensorDescriptor_t yDesc,
                                void *y)
{
//    return HIPDNN_STATUS_NOT_SUPPORTED;
  
    int8_t* devptr = 0;
        
    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;
    
    retVal = hipTomiopenLRNMode(lrnMode, &mimode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    if( sDescToWorkspaceLRN.find((miopenTensorDescriptor_t)yDesc) != sDescToWorkspaceLRN.end() )
    {
        //HGSOS looks like the yDesc is used for the workspace, not the hipdnnLRNDescriptor_t
        
        miStat = miopenLRNGetWorkSpaceSize( (miopenTensorDescriptor_t)yDesc, &workSpaceSize);
        if(  miStat == miopenStatusSuccess )
        {
            hipMalloc( (void**)&devptr, workSpaceSize);
            sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc] = devptr;
            sDescToWorkspaceLRNSize[(miopenTensorDescriptor_t)yDesc] = workSpaceSize;
        }
        else
        {
            return miopenTohipdnnStatus(miStat);
        }
        
    }
    else
    {
        devptr = sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc];
        workSpaceSize = sDescToWorkspaceLRNSize[(miopenTensorDescriptor_t)yDesc];
    }

    return miopenTohipdnnStatus  ( 
                        miopenLRNForward( (miopenHandle_t)handle,
                                            (miopenLRNDescriptor_t)  normDesc,
                                            alpha,
                                            (miopenTensorDescriptor_t)xDesc,
                                            x, 
                                            beta,
                                            (miopenTensorDescriptor_t)yDesc,
                                            y, 
                                            false, // bool do_backward, //HGSOS
                                            devptr));  //HGSOS //NOTYET no workspace size
}



hipdnnStatus_t
hipdnnLRNCrossChannelForwardEx( hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode,  
                                const void* alpha,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x,
                                const void *beta,
                                const hipdnnTensorDescriptor_t yDesc,
                                void *y,
                                size_t workspaceSize,
                                void* workspace)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;
    
    retVal = hipTomiopenLRNMode(lrnMode, &mimode);
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    //mimode is otherwise unused.
    
    return miopenTohipdnnStatus( 
                    miopenLRNForward(  (miopenHandle_t)handle,
                                        (miopenLRNDescriptor_t)  normDesc,
                                        alpha,
                                        (miopenTensorDescriptor_t)xDesc,
                                        x, 
                                        beta,
                                        (miopenTensorDescriptor_t)yDesc,
                                        y, 
                                        false, //bool do_backward, //HGSOS //NOTYET
                                        workspace) );  //NOTYET  no workspace size!
}

    

//=============================================================================

hipdnnStatus_t
hipdnnLRNCrossChannelBackward(  hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode, 
                                const void* alpha,
                                const hipdnnTensorDescriptor_t yDesc,
                                const void *y,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x, const void *beta,
                                const hipdnnTensorDescriptor_t dxDesc,
                                void *dx)
{
//    return HIPDNN_STATUS_NOT_SUPPORTED;
  
    int8_t* devptr = 0;
        
    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;
    
    retVal = hipTomiopenLRNMode(lrnMode, &mimode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    if( sDescToWorkspaceLRN.find((miopenTensorDescriptor_t)yDesc) != sDescToWorkspaceLRN.end() )
    {
        //HGSOS looks like the yDesc is used for the workspace, not the hipdnnLRNDescriptor_t
        
        miStat = miopenLRNGetWorkSpaceSize((miopenTensorDescriptor_t) yDesc, &workSpaceSize);
        if(  miStat == miopenStatusSuccess )
        {
            hipMalloc( (void**)&devptr, workSpaceSize);
            sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc] = devptr;
            sDescToWorkspaceLRNSize[(miopenTensorDescriptor_t)yDesc] = workSpaceSize;
        }
        else
        {
            return miopenTohipdnnStatus(miStat);
        }
        
    }
    else
    {
        devptr = sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc];
        workSpaceSize = sDescToWorkspaceLRNSize[(miopenTensorDescriptor_t)yDesc];
    }

    return hipdnnLRNCrossChannelBackwardEx((miopenHandle_t)handle,
                                            (miopenLRNDescriptor_t)  normDesc,
                                            lrnMode, 
                                            alpha,
                                            (miopenTensorDescriptor_t)yDesc,
                                            y,
                                            (miopenTensorDescriptor_t)dyDesc,
                                            dy,
                                            (miopenTensorDescriptor_t)xDesc,
                                            x, 
                                            beta,
                                            (miopenTensorDescriptor_t)dxDesc,
                                            dx,
                                            workSpaceSize, 
                                            devptr);
}

hipdnnStatus_t
hipdnnLRNCrossChannelBackwardEx(    hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode, 
                                const void* alpha,
                                const hipdnnTensorDescriptor_t yDesc,
                                const void *y,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x, const void *beta,
                                const hipdnnTensorDescriptor_t dxDesc,
                                void *dx,
                                size_t workspacesize, //HGSOS //NOTYET unused!!!
                                void* workspace)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;
    
    retVal = hipTomiopenLRNMode(lrnMode, &mimode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    //mimode is otherwise unused.
    return miopenTohipdnnStatus( 
                miopenLRNBackward( (miopenHandle_t)handle,
                                    (miopenLRNDescriptor_t)  normDesc,
                                    alpha,
                                    (miopenTensorDescriptor_t)yDesc,
                                    y,
                                    (miopenTensorDescriptor_t)dyDesc,
                                    dy,
                                    (miopenTensorDescriptor_t)xDesc,
                                    x, beta,
                                    (miopenTensorDescriptor_t)dxDesc,
                                    dx, workspace));
}


//==================================!

hipdnnStatus_t
hipdnnDeriveBNTensorDescriptor( hipdnnTensorDescriptor_t derivedBnDesc,
                                const hipdnnTensorDescriptor_t xDesc,
                                hipdnnBatchNormMode_t mode)
{
    return  miopenTohipdnnStatus(
                miopenDeriveBNTensorDescriptor( (miopenTensorDescriptor_t)derivedBnDesc,
                                                (miopenTensorDescriptor_t)xDesc,
                                                hipTomiopenBatchNormMode(mode)));
}

//=============================================================================

hipdnnStatus_t
hipdnnBatchNormalizationForwardTraining(    hipdnnHandle_t handle,
                                            hipdnnBatchNormMode_t mode,
                                            void *alpha, 
                                            void *beta,
                                            const hipdnnTensorDescriptor_t xDesc,
                                            const void *x,
                                            const hipdnnTensorDescriptor_t yDesc,
                                            void *y,
                                            const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                            void *bnScale, void *bnBias,
                                            double exponentialAverageFactor,
                                            void *resultRunningMean,
                                            void *resultRunningVariance,
                                            double epsilon,
                                            void *resultSaveMean,
                                            void *resultSaveInvVariance)
{
    return  miopenTohipdnnStatus(
                miopenBatchNormalizationForwardTraining(    
                                           (miopenHandle_t)handle,
                                            hipTomiopenBatchNormMode(mode),
                                            alpha, 
                                            beta,
                                            (miopenTensorDescriptor_t)xDesc,
                                            x,
                                            (miopenTensorDescriptor_t)yDesc,
                                            y,
                                            (miopenTensorDescriptor_t)bnScaleBiasMeanVarDesc,
                                            bnScale, bnBias,
                                            exponentialAverageFactor,
                                            resultRunningMean,
                                            resultRunningVariance,
                                            epsilon,
                                            resultSaveMean,
                                            resultSaveInvVariance));
}
//=============================================================================

hipdnnStatus_t
hipdnnnBatchNormalizationForwardInference(  hipdnnHandle_t handle,
                                            hipdnnBatchNormMode_t mode,
                                            void *alpha, 
                                            void *beta,
                                            const hipdnnTensorDescriptor_t xDesc,
                                            const void *x,
                                            const hipdnnTensorDescriptor_t yDesc,
                                            void *y,
                                            const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                            const void *bnScale, 
                                            const void *bnBias,
                                            const void *estimatedMean,
                                            const void *estimatedVariance,
                                            double epsilon)
{
    
    return HIPDNN_STATUS_NOT_SUPPORTED;
    
#ifdef NOTYET
    //arguments 10,11,12,13 below are not const in miopen. 
return  miopenTohipdnnStatus(
                miopenBatchNormalizationForwardInference(  (miopenHandle_t)handle,
                                                            hipTomiopenBatchNormMode(mode),
                                                            alpha, 
                                                            beta,
                                                            (miopenTensorDescriptor_t)xDesc,
                                                            x,
                                                            (miopenTensorDescriptor_t)yDesc,
                                                            y,
                                                            (miopenTensorDescriptor_t)bnScaleBiasMeanVarDesc,
                                                            bnScale, bnBias,
                                                            estimatedMean,
                                                            estimatedVariance,
                                                            epsilon));
#endif

}
//=============================================================================

hipdnnStatus_t
hipdnnBatchNormalizationBackward(hipdnnHandle_t handle,
                                hipdnnBatchNormMode_t mode,
                                const void *alphaDataDiff,
                                const void *betaDataDiff,
                                const void *alphaParamDiff,
                                const void *betaParamDiff,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnTensorDescriptor_t dxDesc,
                                void *dx,
                                const hipdnnTensorDescriptor_t bnScaleBiasDiffDesc,
                                const void *bnScale,
                                void *resultBnScaleDiff,
                                void *resultBnBiasDiff,
                                double epsilon,
                                const void *savedMean,
                                const void *savedInvVariance)
{
return  miopenTohipdnnStatus(
            miopenBatchNormalizationBackward(  (miopenHandle_t)handle,
                                                hipTomiopenBatchNormMode(mode),
                                                alphaDataDiff,
                                                betaDataDiff,
                                                alphaParamDiff,
                                                betaParamDiff,
                                                (miopenTensorDescriptor_t)xDesc,
                                                x,
                                                (miopenTensorDescriptor_t)dyDesc,
                                                dy,
                                                (miopenTensorDescriptor_t)dxDesc,
                                                dx,
                                                (miopenTensorDescriptor_t)bnScaleBiasDiffDesc,
                                                bnScale,
                                                resultBnScaleDiff,
                                                resultBnBiasDiff,
                                                epsilon,
                                                savedMean,
                                                savedInvVariance));
}

hipdnnStatus_t hipdnnSetTensorNdDescriptor(
                                hipdnnTensorDescriptor_t             tensorDesc,
                                hipdnnDataType_t                     dataType,
                                int                                 nbDims,
                                const int                           dimA[],
                                const int                           strideA[] )
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t moDT;
    retVal = hipTomiopenDataType(dataType, &moDT);
    if(retVal == HIPDNN_STATUS_SUCCESS)
         return miopenTohipdnnStatus(miopenSetTensorDescriptor((miopenTensorDescriptor_t)tensorDesc, moDT, nbDims, const_cast<int*>(dimA), const_cast<int*>(strideA)));
    return retVal;
}

hipdnnStatus_t hipdnnGetTensorNdDescriptor(
                                const hipdnnTensorDescriptor_t       tensorDesc,
                                int                                 nbDimsRequested,
                                hipdnnDataType_t                    *dataType,
                                int                                *nbDims,
                                int                                 dimA[],
                                int                                 strideA[] )
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t moDT;
    retVal = miopenTohipdnnStatus(miopenGetTensorDescriptor((miopenTensorDescriptor_t)tensorDesc, &moDT, dimA, strideA));
    if(retVal == HIPDNN_STATUS_SUCCESS)
    {
         retVal = miopenTohipDataType(moDT, dataType);
         if(retVal == HIPDNN_STATUS_SUCCESS)
             retVal = miopenTohipdnnStatus(miopenGetTensorDescriptorSize((miopenTensorDescriptor_t)tensorDesc, nbDims));
    }
    return retVal;
}

hipdnnStatus_t hipdnnSetFilterNdDescriptor(
                                hipdnnFilterDescriptor_t             filterDesc,
                                hipdnnDataType_t                     dataType, // image data type
                                hipdnnTensorFormat_t                 format,
                                int                                 nbDims,
                                const int                           filterDimA[] )
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t moDT;
    retVal = hipTomiopenDataType(dataType, &moDT);
    if(retVal == HIPDNN_STATUS_SUCCESS)
         return miopenTohipdnnStatus(miopenSetTensorDescriptor((miopenTensorDescriptor_t)filterDesc, moDT, nbDims, const_cast<int*>(filterDimA), const_cast<int*>(filterDimA)));
    return retVal;	
}

hipdnnStatus_t hipdnnGetFilterNdDescriptor(
                                const hipdnnFilterDescriptor_t       filterDesc,
                                int                                 nbDimsRequested,
                                hipdnnDataType_t                    *dataType, // image data type
                                hipdnnTensorFormat_t                *format,
                                int                                *nbDims,
                                int                                 filterDimA[] )
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t moDT;
    retVal = miopenTohipdnnStatus(miopenGetTensorDescriptor((miopenTensorDescriptor_t)filterDesc, &moDT, filterDimA, filterDimA));
    if(retVal == HIPDNN_STATUS_SUCCESS)
    {
         retVal = miopenTohipDataType(moDT, dataType);
         if(retVal == HIPDNN_STATUS_SUCCESS)
         {
             retVal = miopenTohipdnnStatus(miopenGetTensorDescriptorSize((miopenTensorDescriptor_t)filterDesc, nbDims));
             *format = HIPDNN_TENSOR_NCHW; //miopen defines only this format
         } 
    }
    return retVal;
}


hipdnnStatus_t hipdnnDestroyFilterDescriptor(
                                hipdnnFilterDescriptor_t             filterDesc )
{
    return miopenTohipdnnStatus(miopenDestroyTensorDescriptor((miopenTensorDescriptor_t)filterDesc));
}

//RNN APIs

hipdnnStatus_t hipdnnCreateRNNDescriptor(hipdnnRNNDescriptor_t * rnnDesc)
{
        return miopenTohipdnnStatus(miopenCreateRNNDescriptor(reinterpret_cast<miopenRNNDescriptor_t *>(rnnDesc)));
}

hipdnnStatus_t hipdnnDestroyRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc)
{
        return miopenTohipdnnStatus(miopenDestroyRNNDescriptor((miopenRNNDescriptor_t)rnnDesc));
}

hipdnnStatus_t  hipdnnCreatePersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                             const int minibatch,
                                             const hipdnnDataType_t dataType,
                                             hipdnnPersistentRNNPlan_t * plan)
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}
                                             
hipdnnStatus_t  hipdnnSetPersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                          hipdnnPersistentRNNPlan_t plan)
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}
                                          
hipdnnStatus_t  hipdnnDestroyPersistentRNNPlan(hipdnnPersistentRNNPlan_t plan)
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}
                                          
hipdnnStatus_t  hipdnnSetRNNDescriptor_v6(hipdnnHandle_t handle, 
                                                hipdnnRNNDescriptor_t rnnDesc,
                                                const int hiddenSize, 
                                                const int numLayers, 
                                                hipdnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
                                                hipdnnRNNInputMode_t inputMode,                                                 
                                                hipdnnDirectionMode_t direction, 
                                                hipdnnRNNMode_t mode, 
                                                hipdnnRNNAlgo_t algo, 
                                                hipdnnDataType_t dataType)
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}


hipdnnStatus_t  hipdnnSetRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc,
                                                int hiddenSize, 
                                                int numLayers, 
                                                hipdnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
                                                hipdnnRNNInputMode_t inputMode,                                                 
                                                hipdnnDirectionMode_t direction, 
                                                hipdnnRNNMode_t mode, 
                                                hipdnnDataType_t dataType)
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t  hipdnnGetRNNWorkspaceSize( hipdnnHandle_t              handle,
                                                    const hipdnnRNNDescriptor_t rnnDesc,  
                                                    const int seqLength, 
                                                    const hipdnnTensorDescriptor_t    *xDesc,
                                                    size_t                     *sizeInBytes
                                                    )
{
        return miopenTohipdnnStatus(miopenGetRNNWorkspaceSize((miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength, reinterpret_cast<miopenTensorDescriptor_t*>(const_cast<hipdnnTensorDescriptor_t*>(xDesc)), sizeInBytes));
}
                                                      
hipdnnStatus_t  hipdnnGetRNNTrainingReserveSize( hipdnnHandle_t              handle,
                                                          const hipdnnRNNDescriptor_t rnnDesc,  
                                                          const int seqLength, 
                                                          const hipdnnTensorDescriptor_t    *xDesc,
                                                          size_t                     *sizeInBytes
                                                    )
{
        return miopenTohipdnnStatus(miopenGetRNNTrainingReserveSize((miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength, reinterpret_cast<miopenTensorDescriptor_t *>(const_cast<hipdnnTensorDescriptor_t*>(xDesc)), sizeInBytes));
}

                                                    
hipdnnStatus_t  hipdnnGetRNNParamsSize( hipdnnHandle_t              handle,
                                                 const hipdnnRNNDescriptor_t rnnDesc,  
                                                 const hipdnnTensorDescriptor_t    xDesc,                                                    
                                                 size_t                     *sizeInBytes,
                                                 hipdnnDataType_t dataType
                                                    )
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t  hipdnnGetRNNLinLayerMatrixParams( hipdnnHandle_t              handle,
                             const hipdnnRNNDescriptor_t rnnDesc,  
                             const int layer,
                             const hipdnnTensorDescriptor_t xDesc, 
                             const hipdnnFilterDescriptor_t wDesc, 
                             const void * w, 
                             const int linLayerID,  
                             hipdnnFilterDescriptor_t linLayerMatDesc, 
                             void ** linLayerMat
                             )
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t  hipdnnGetRNNLinLayerBiasParams( hipdnnHandle_t              handle,
                             const hipdnnRNNDescriptor_t rnnDesc,  
                             const int layer,
                             const hipdnnTensorDescriptor_t xDesc, 
                             const hipdnnFilterDescriptor_t wDesc, 
                             const void * w, 
                             const int linLayerID, 
                             hipdnnFilterDescriptor_t linLayerBiasDesc, 
                             void ** linLayerBias                       
                             )
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t  hipdnnRNNForwardInference( hipdnnHandle_t handle, 
                                                    const hipdnnRNNDescriptor_t rnnDesc, 
                                                    const int seqLength, 
                                                    const hipdnnTensorDescriptor_t * xDesc, 
                                                    const void * x, 
                                                    const hipdnnTensorDescriptor_t hxDesc, 
                                                    const void * hx, 
                                                    const hipdnnTensorDescriptor_t cxDesc, 
                                                    const void * cx, 
                                                    const hipdnnFilterDescriptor_t wDesc, 
                                                    const void * w, 
                                                    const hipdnnTensorDescriptor_t *yDesc,  
                                                    void * y, 
                                                    const hipdnnTensorDescriptor_t hyDesc, 
                                                    void * hy, 
                                                    const hipdnnTensorDescriptor_t cyDesc, 
                                                    void * cy, 
                                                    void * workspace, 
                                                    size_t workSpaceSizeInBytes)
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t  hipdnnRNNForwardTraining( hipdnnHandle_t handle, 
                                                   const hipdnnRNNDescriptor_t rnnDesc, 
                                                   const int seqLength, 
                                                   const hipdnnTensorDescriptor_t *xDesc, 
                                                   const void * x, 
                                                   const hipdnnTensorDescriptor_t hxDesc, 
                                                   const void * hx, 
                                                   const hipdnnTensorDescriptor_t cxDesc, 
                                                   const void * cx, 
                                                   const hipdnnFilterDescriptor_t wDesc, 
                                                   const void * w, 
                                                   const hipdnnTensorDescriptor_t *yDesc,  
                                                   void * y, 
                                                   const hipdnnTensorDescriptor_t hyDesc, 
                                                   void * hy, 
                                                   const hipdnnTensorDescriptor_t cyDesc, 
                                                   void * cy, 
                                                   void * workspace, 
                                                   size_t workSpaceSizeInBytes,
                                                   void * reserveSpace, 
                                                   size_t reserveSpaceSizeInBytes)
{
        return miopenTohipdnnStatus(miopenRNNForwardTraining((miopenHandle_t)handle, (miopenRNNDescriptor_t) rnnDesc, seqLength, reinterpret_cast<miopenTensorDescriptor_t *>(const_cast<hipdnnTensorDescriptor_t*>(xDesc)), x, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(hxDesc), hx, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(cxDesc), cx, (miopenTensorDescriptor_t)wDesc, w, reinterpret_cast<miopenTensorDescriptor_t *>(const_cast<hipdnnTensorDescriptor_t*>(yDesc)), y, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(hyDesc), hy,(miopenTensorDescriptor_t) const_cast<hipdnnTensorDescriptor_t>(cyDesc), cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));
}

hipdnnStatus_t  hipdnnRNNBackwardData( hipdnnHandle_t handle, 
                                                const hipdnnRNNDescriptor_t rnnDesc, 
                                                const int seqLength, 
                                                const hipdnnTensorDescriptor_t * yDesc, 
                                                const void * y,                                                
                                                const hipdnnTensorDescriptor_t * dyDesc, 
                                                const void * dy, 
                                                const hipdnnTensorDescriptor_t dhyDesc, 
                                                const void * dhy, 
                                                const hipdnnTensorDescriptor_t dcyDesc, 
                                                const void * dcy, 
                                                const hipdnnFilterDescriptor_t wDesc, 
                                                const void * w, 
                                                const hipdnnTensorDescriptor_t hxDesc, 
                                                const void * hx,                                                                  
                                                const hipdnnTensorDescriptor_t cxDesc, 
                                                const void * cx,                                                 
                                                const hipdnnTensorDescriptor_t * dxDesc, 
                                                void * dx, 
                                                const hipdnnTensorDescriptor_t dhxDesc,
                                                void * dhx,
                                                const hipdnnTensorDescriptor_t dcxDesc,
                                                void * dcx,
                                                void * workspace,
                                                size_t workSpaceSizeInBytes,
                                                void * reserveSpace, 
                                                size_t reserveSpaceSizeInBytes )
{
        return miopenTohipdnnStatus(miopenRNNBackwardData((miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength, reinterpret_cast<miopenTensorDescriptor_t *>(const_cast<hipdnnTensorDescriptor_t*>(yDesc)), y, reinterpret_cast<miopenTensorDescriptor_t *>(const_cast<hipdnnTensorDescriptor_t*>(dyDesc)), dy, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(dhyDesc), dhy, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(dcyDesc), dcy, (miopenTensorDescriptor_t)wDesc, w, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(hxDesc), hx, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(cxDesc), cx, reinterpret_cast<miopenTensorDescriptor_t *>(const_cast<hipdnnTensorDescriptor_t*>(dxDesc)), dx, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(dhxDesc), dhx, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(dcxDesc), dcx, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));
}


hipdnnStatus_t  hipdnnRNNBackwardWeights( hipdnnHandle_t handle, 
                                                   const hipdnnRNNDescriptor_t rnnDesc, 
                                                   const int seqLength, 
                                                   const hipdnnTensorDescriptor_t * xDesc, 
                                                   const void * x, 
                                                   const hipdnnTensorDescriptor_t hxDesc, 
                                                   const void * hx,                                                   
                                                   const hipdnnTensorDescriptor_t * yDesc, 
                                                   const void * y,
                                                   const void * workspace, 
                                                   size_t workSpaceSizeInBytes, 
                                                   const hipdnnFilterDescriptor_t dwDesc, 
                                                   void * dw,
                                                   const void * reserveSpace, 
                                                   size_t reserveSpaceSizeInBytes )
{
        return miopenTohipdnnStatus(miopenRNNBackwardWeights((miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength, reinterpret_cast<miopenTensorDescriptor_t *>(const_cast<hipdnnTensorDescriptor_t*>(xDesc)), x, (miopenTensorDescriptor_t)const_cast<hipdnnTensorDescriptor_t>(hxDesc), hx, reinterpret_cast<miopenTensorDescriptor_t *>(const_cast<hipdnnTensorDescriptor_t*>(yDesc)), y, (miopenTensorDescriptor_t)dwDesc, dw, const_cast<void*>(workspace), workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));
}

hipdnnStatus_t hipdnnSetPoolingNdDescriptor(hipdnnPoolingDescriptor_t poolingDesc,
                                            const hipdnnPoolingMode_t mode,
                                            const hipdnnNanPropagation_t maxpoolingNanOpt,
                                            int nbDims,
                                            const int windowDimA[],
                                            const int paddingA[],
                                            const int strideA[] )
{
	return HIPDNN_STATUS_NOT_SUPPORTED;
}

const char * hipdnnGetErrorString(hipdnnStatus_t status)
{
	return "Not implemented";
}

hipdnnStatus_t hipdnnSetConvolutionNdDescriptor(
                                        hipdnnConvolutionDescriptor_t convDesc,
                                        int arrayLength, /* nbDims-2 size */
                                        const int padA[],
                                        const int filterStrideA[],
                                        const int dilationA[],
                                        hipdnnConvolutionMode_t mode,
                                        hipdnnDataType_t computeType ) // convolution data type
{
        if(arrayLength == 2)
	{
		int pad_h, pad_w, u, v;
		pad_h = padA[0];
		pad_w = padA[1];
		u = filterStrideA[0];
		v = filterStrideA[1];
		hipdnnStatus_t retVal = HIPDNN_STATUS_NOT_SUPPORTED;
		retVal = miopenTohipdnnStatus(miopenInitConvolutionDescriptor((miopenConvolutionDescriptor_t)convDesc, miopenConvolution, pad_h, pad_w, u, v, 1, 1));
		return retVal;	
	}
	else
        	return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnBatchNormalizationForwardInference(
                                hipdnnHandle_t                       handle,
                                hipdnnBatchNormMode_t                mode,
                                const void                         *alpha, // alpha[0] = result blend factor
                                const void                         *beta,  // beta[0] = dest layer blend factor
                                const hipdnnTensorDescriptor_t       xDesc,
                                const void                         *x,     // NxCxHxW
                                const hipdnnTensorDescriptor_t       yDesc,
                                void                               *y,     // NxCxHxW
                                const hipdnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,
                                const void                         *bnScale,
                                const void                         *bnBias,
                                const void                         *estimatedMean,
                                const void                         *estimatedVariance,
                                double                              epsilon )
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnCreateDropoutDescriptor(hipdnnDropoutDescriptor_t * dropoutDesc)
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc,
                                                    hipdnnHandle_t handle,
                                                    float dropout,
                                                    void * states,
                                                    size_t stateSizeInBytes,
                                                    unsigned long long seed)
{
        return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDropoutGetStatesSize(hipdnnHandle_t handle, size_t * sizeInBytes)
{
	return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc)
{
	return HIPDNN_STATUS_NOT_SUPPORTED;
}
