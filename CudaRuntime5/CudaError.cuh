#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if(err != cudaSuccess) { \
std::cerr<<"cuda error "<<cudaGetErrorString(err)<<" at "<<__FILE__<<":"<<__LINE__<<std::endl; \
exit(EXIT_FAILURE); \
} \
} while(0)