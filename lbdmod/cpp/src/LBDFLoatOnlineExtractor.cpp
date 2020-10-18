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

#include "../include/LBDFLoatOnlineExtractor.h"

LBDFLoatOnlineExtractor::LBDFLoatOnlineExtractor()
{
    cv::line_descriptor::BinaryDescriptor::Params p;
    p.numOfOctave_ = n_octaves;
    p.widthOfBand_ = 7;
    p.ksize_ = 5;
    p.factor = pyramid_factor;

    bd = cv::line_descriptor::BinaryDescriptor(p);
}

void LBDFLoatOnlineExtractor::ExtractDetections(const cv::Mat& frame, std::vector<KeyLine> *extrLines)
{

    cv::line_descriptor::BinaryDescriptor::Params p;
    p.numOfOctave_ = n_octaves;
    p.widthOfBand_ = 7;
    p.ksize_ = 5;
    p.factor = pyramid_factor;

    bd = cv::line_descriptor::BinaryDescriptor(p);
    std::vector<KeyLine> lines;
    bd.detect(frame, lines, cv::Mat());
    *extrLines = lines;
}

void LBDFLoatOnlineExtractor::ExtractDescriptors(const cv::Mat &frame, std::vector<KeyLine> &keyLines,
                                                 cv::Mat *lineDescs)
{
    cv::Mat descs_loc;
    bd.compute(frame, keyLines, descs_loc, true);
    descs_loc.copyTo(*lineDescs);
}