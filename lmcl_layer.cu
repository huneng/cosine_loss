#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/lmcl_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LMCLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype scale;
    // norm weight
    if(this->phase_ == TRAIN){
        Dtype *weight = this->blobs_[0]->mutable_gpu_data();

        scale = Dtype(0.0f);
        for(int i = 0; i < N_; i ++){
            Dtype dot;
            caffe_gpu_dot(K_, weight, weight, &dot);
            dot = sqrt(dot + 1e-10);
            caffe_gpu_scal(K_, Dtype(1.0) / dot, weight);
            weight += K_;

            scale += dot;
        }

        scale /= N_;

        caffe_gpu_scal(N_ * K_, scale, this->blobs_[0]->mutable_gpu_data());
    }

    // y = W * x
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, M_, N_, K_,
            Dtype(1.0), bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), 
            Dtype(0.0), top[0]->mutable_gpu_data());

    //*
    // margin
    if(this->phase_ == TRAIN && margin_ > Dtype(1e-5f)){
        Dtype *y = top[0]->mutable_cpu_data();
        const Dtype *ptrLabel = bottom[1]->cpu_data();

        for(int i = 0; i < M_; i ++){
            int label = int(ptrLabel[i]);
            CHECK(label < N_);
            y[label] = ((y[label] / scale) - margin_) * scale;
            y += N_;
        }
    }
    // */
}


template <typename Dtype>
void LMCLLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
    if (this->param_propagate_down_[0]) {
        const Dtype* dy = top[0]->gpu_diff();
        const Dtype* x = bottom[0]->gpu_data();
        Dtype *dw = this->blobs_[0]->mutable_gpu_diff();

        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                N_, K_, M_,
                (Dtype)1., dy, x, (Dtype)0., dw);
    }

    if (propagate_down[0]) {
        const Dtype* dy = top[0]->gpu_diff();
        const Dtype* w = this->blobs_[0]->gpu_data();
        Dtype *dx = bottom[0]->mutable_gpu_diff();

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                M_, K_, N_,
                (Dtype)1., dy, w, (Dtype)0., dx);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(LMCLLayer);

}  // namespace caffe
