/**
* This file is part of LLD-SLAM.
*
* Copyright (C) 2018 Alexander Vakhitov <alexander.vakhitov at gmail dot com> (Skoltech)
* For more information see <https://github.com/alexandervakhitov/lld-slam>
*
* lld-slam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* lld-slam is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LLD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LBD_MOD_LBD_MOD_FUNCS_H
#define LBD_MOD_LBD_MOD_FUNCS_H

#include "../src/precomp.hpp"

void DetectEDLines(const cv::Mat& img, int n_octaves, double factor, std::vector<cv::line_descriptor::KeyLine>* detections_p);

void DetectComputeLBD(const cv::Mat& image, int n_octaves, double factor, std::vector<cv::line_descriptor::KeyLine>* detections, cv::Mat* descs_p);

void ComputeLBD(const cv::Mat& image, int n_octaves, double factor, std::vector<cv::line_descriptor::KeyLine>& detections_p, cv::Mat* descs_p);

void MatchLBD(const cv::Mat& descs1, const cv::Mat& descs2, std::vector<cv::DMatch>* matches_p);

#endif //LBD_MOD_LBD_MOD_FUNCS_H
