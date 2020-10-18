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

#ifndef ORB_SLAM2_LINEEXTRACTOR_H
#define ORB_SLAM2_LINEEXTRACTOR_H

#include "lbdmod/line_descriptor.hpp"

typedef cv::line_descriptor::KeyLine KeyLine;

class LineExtractor {
public:

    void ExtractLines(const cv::Mat& frame, std::vector<KeyLine>* extrLines, cv::Mat* lineDescs);

    virtual void ExtractDescriptors(const cv::Mat& frame, std::vector<KeyLine>& keyLines, cv::Mat* lineDescs) = 0;
    virtual void ExtractDetections(const cv::Mat& frame, std::vector<KeyLine>* extrLines) = 0;

    virtual ~LineExtractor(){};

};


#endif //ORB_SLAM2_LINEEXTRACTOR_H
