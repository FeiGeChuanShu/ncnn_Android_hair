// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
    colorFlag=0;
}

int NanoDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    hairseg.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    hairseg.opt = ncnn::Option();

#if NCNN_VULKAN
    hairseg.opt.use_vulkan_compute = use_gpu;
#endif

    hairseg.opt.num_threads = ncnn::get_big_cpu_count();
    hairseg.opt.blob_allocator = &blob_pool_allocator;
    hairseg.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    hairseg.load_param(parampath);
    hairseg.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    hairseg.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    hairseg.opt = ncnn::Option();

#if NCNN_VULKAN
    hairseg.opt.use_vulkan_compute = use_gpu;
#endif
    hairseg.opt.lightmode = true;
    hairseg.opt.num_threads = ncnn::get_big_cpu_count();
    hairseg.opt.blob_allocator = &blob_pool_allocator;
    hairseg.opt.workspace_allocator = &workspace_pool_allocator;
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    hairseg.load_param(mgr,parampath);
    hairseg.load_model(mgr,modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}


void NanoDet::seg(cv::Mat &rgb,cv::Mat &mask,cv::Rect &box)
{
    cv::Mat faceRoiImage = rgb.clone();
    ncnn::Extractor ex_hair = hairseg.create_extractor();
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(faceRoiImage.data,
        ncnn::Mat::PIXEL_RGB, faceRoiImage.cols, faceRoiImage.rows,target_size,target_size);

    ncnn_in.substract_mean_normalize(mean_vals, norm_vals);
    ex_hair.input("input",ncnn_in);
    ncnn::Mat out;
    ex_hair.extract("1006",out);
    float *scoredata = (float*)out.data;

    mask = cv::Mat(target_size, target_size, CV_32FC1,scoredata);
}
int NanoDet::draw(cv::Mat& rgb)
{
    static const unsigned  char part_colors[8][3] = {{0, 0, 255}, {255, 85, 0}, {255, 170, 0},
                   {255, 0, 85}, {255, 0, 170},
                   {0, 255, 0}, {170, 255, 255}, {255, 255, 255}};
    int color_index = 0;

    cv::Mat mask;
    cv::Rect  box;
    seg(rgb,mask,box);
    cv::Mat maskResize;
    cv::resize(mask,maskResize,cv::Size(rgb.cols,rgb.rows),0,0,1);

    for(size_t h = 0; h < maskResize.rows; h++)
    {
        cv::Vec3b* pRgb = rgb.ptr<cv::Vec3b >(h);
        float *alpha = maskResize.ptr<float>(h);
        for(size_t w = 0; w < maskResize.cols; w++)
        {
            float weight = alpha[w];
            pRgb[w] = cv::Vec3b(part_colors[colorFlag][2]*weight+pRgb[w][0]*(1-weight),
                                    part_colors[colorFlag][1]*weight+pRgb[w][1]*(1-weight),
                                    part_colors[colorFlag][0]*weight+pRgb[w][2]*(1-weight));
        }
    }
    if(colorFlag < 7)
        colorFlag++;
    else
        colorFlag=0;
    return 0;
}
