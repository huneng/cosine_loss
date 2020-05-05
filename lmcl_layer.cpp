#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/lmcl_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LMCLLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    margin_ = this->layer_param_.lmcl_param().margin();
    axis_   = bottom[0]->CanonicalAxisIndex(this->layer_param_.lmcl_param().axis());
    N_      = this->layer_param_.lmcl_param().num_output();
    K_      = bottom[0]->count(axis_);

    CHECK(margin_ >= 0.0f && margin_ < 0.5f);

    if(margin_ > 0){
        CHECK(bottom.size() == 2);
        CHECK(bottom[1]->num() == bottom[1]->count());
    }

    if (this->blobs_.size() == 0) {
        vector<int> weight_shape(2);

        weight_shape[0] = N_;
        weight_shape[1] = K_;

        this->blobs_.resize(1);
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

        // fill the weights
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                    this->layer_param_.lmcl_param().weight_filler()));

        weight_filler->Fill(this->blobs_[0].get());
    }
    else {
        LOG(INFO) << "Skipping parameter initialization";
    }

    this->param_propagate_down_.resize(this->blobs_.size(), true);
}


template <typename Dtype>
void LMCLLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    const int new_K = bottom[0]->count(axis_);

    CHECK_EQ(K_, new_K)
        << "Input size incompatible with inner product parameters.";

    M_ = bottom[0]->count(0, axis_);

    vector<int> top_shape = bottom[0]->shape();

    top_shape.resize(axis_ + 1);
    top_shape[axis_] = N_;
    top[0]->Reshape(top_shape);
}


template <typename Dtype>
void LMCLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    Dtype scale;
    // norm weight
    if(this->phase_ == TRAIN){
        Dtype *weight = this->blobs_[0]->mutable_cpu_data();

        scale = Dtype(0.0f);
        for(int i = 0; i < N_; i ++){
            Dtype dot;
            dot = caffe_cpu_dot(K_, weight, weight);
            dot = sqrt(dot + 1e-10);
            caffe_scal(K_, Dtype(1.0) / dot, weight);
            weight += K_;

            scale += dot;
        }

        scale /= N_;

        caffe_scal(N_ * K_, scale, this->blobs_[0]->mutable_cpu_data());
    }

    // y = W * x
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, M_, N_, K_,
            Dtype(1.0), bottom[0]->cpu_data(), this->blobs_[0]->cpu_data(),
            Dtype(0.0), top[0]->mutable_cpu_data());

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
void LMCLLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
    if (this->param_propagate_down_[0]) {
        const Dtype* dy = top[0]->cpu_diff();
        const Dtype* x = bottom[0]->cpu_data();
        Dtype *dw = this->blobs_[0]->mutable_cpu_diff();

        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                N_, K_, M_,
                (Dtype)1., dy, x, (Dtype)0., dw);
    }

    if (propagate_down[0]) {
        const Dtype* dy = top[0]->cpu_diff();
        const Dtype* w = this->blobs_[0]->cpu_data();
        Dtype *dx = bottom[0]->mutable_cpu_diff();

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                M_, K_, N_,
                (Dtype)1., dy, w, (Dtype)0., dx);
    }
}

#ifdef CPU_ONLY
STUB_GPU(LMCLLayer);
#endif

INSTANTIATE_CLASS(LMCLLayer);
REGISTER_LAYER_CLASS(LMCL);

}  // namespace caffe
