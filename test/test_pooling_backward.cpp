#include "test_pooling_backward.hpp"

TEST(pooling_backward, func_check_pooling_stride_2x2) {
  float avg_time = 0;
  int oheight = 4, owidth = 4;
  pool_bwd test_case(1, 1, 4, 4, 2, 2, 0, 0, 2, 2, 1, 1, oheight, owidth);
  Memory<float> dataSrc(16);
  Memory<float> dataGrad(16);
  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);
  Memory<float> dataDst(test_case.on * test_case.ochannel * test_case.oheight *
                        test_case.owidth);

  int ip_size[4] = {1,1,4,4};
  int k_size[4] = {1,1,2,2};
  int op_size[4] =  {test_case.on, test_case.ochannel, test_case.oheight, test_case.owidth};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                                  dataDst.gpu(), &avg_time);
    
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

float* temp = dataGrad.getDataFromGPU();

std::string str  = convert_to_string((float*)temp,(int)dataGrad.get_num_elements());

    std::string strt = "./result_unittest.csv";
    std::string testname = "pooling_backward:func_check_pooling_stride_2x2";
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_backward, func_check_pooling_batch32) {
  float avg_time = 0;
  int oheight = 4, owidth = 4;
  pool_bwd test_case(32, 1, 4, 4, 2, 2, 0, 0, 2, 2, 32, 1, oheight, owidth);
  Memory<float> dataSrc(test_case.in * test_case.ichannel * test_case.iheight * test_case.iwidth);
  Memory<float> dataGrad(test_case.in * test_case.ichannel * test_case.iheight * test_case.iwidth);
  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);
  Memory<float> dataDst(test_case.on * test_case.ochannel * test_case.oheight *
                        test_case.owidth);

  int ip_size[4] = {32,1,4,4};
  int k_size[4] = {1,1,2,2};
  int op_size[4] =  {test_case.on, test_case.ochannel, test_case.oheight, test_case.owidth};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                                  dataDst.gpu(), &avg_time);
    
  
float* temp = dataGrad.getDataFromGPU();

std::string str  = convert_to_string((float*)temp,(int)dataGrad.get_num_elements());
 
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "pooling_backward:func_check_pooling_batch32";
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_backward, func_check_pooling_batch64) {
  float avg_time = 0;
  int oheight = 4, owidth = 4;
  pool_bwd test_case(64, 1, 4, 4, 2, 2, 0, 0, 2, 2, 64, 1, oheight, owidth);
  Memory<float> dataSrc(test_case.in * test_case.ichannel * test_case.iheight * test_case.iwidth);
  Memory<float> dataGrad(test_case.in * test_case.ichannel * test_case.iheight * test_case.iwidth);
  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);
  Memory<float> dataDst(test_case.on * test_case.ochannel * test_case.oheight *
                        test_case.owidth);

  int ip_size[4] = {64,1,4,4};
  int k_size[4] = {1,1,2,2};
  int op_size[4] =  {test_case.on, test_case.ochannel, test_case.oheight, test_case.owidth};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                                  dataDst.gpu(), &avg_time);
    
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "pooling_backward:func_check_pooling_batch64";
    float* temp = dataGrad.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_backward, func_check_pooling_batch128) {
  float avg_time = 0;
  int oheight = 4, owidth = 4;
  pool_bwd test_case(128, 1, 4, 4, 2, 2, 0, 0, 2, 2, 128, 1, oheight, owidth);
  Memory<float> dataSrc(test_case.in * test_case.ichannel * test_case.iheight * test_case.iwidth);
  Memory<float> dataGrad(test_case.in * test_case.ichannel * test_case.iheight * test_case.iwidth);
  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);
  Memory<float> dataDst(test_case.on * test_case.ochannel * test_case.oheight *
                        test_case.owidth);

  int ip_size[4] = {128,1,4,4};
  int k_size[4] = {1,1,2,2};
  int op_size[4] =  {test_case.on, test_case.ochannel, test_case.oheight, test_case.owidth};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                                  dataDst.gpu(), &avg_time);
    
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "pooling_backward:func_check_pooling_batch128";
    float* temp = dataGrad.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
}
