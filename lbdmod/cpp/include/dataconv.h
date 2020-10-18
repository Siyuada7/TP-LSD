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

#ifndef PYLBD_DATACONV_H
#define PYLBD_DATACONV_H

#include "opencv2/line_descriptor.hpp"
using namespace cv;
cv::Mat ConvertKeyLines2Mat(const std::vector<cv::line_descriptor::KeyLine>& detections);
std::vector<cv::line_descriptor::KeyLine> ConvertMat2KeyLines(const cv::Mat& lines_data);
cv::Mat ConvertMatches2Mat(const std::vector<cv::DMatch>& matchingVec);
std::vector<cv::DMatch> ConvertMat2Matches(const cv::Mat& matchMat);
cv::Mat KeylineconvertMat(const std::vector<cv::line_descriptor::KeyLine>& detections);
std::vector<cv::line_descriptor::KeyLine> lineconvertKeyline(cv::Mat array, int width=640, int height=480);

struct Line
{
	// Line 的初始化见函数 LineParameters
	Point2f StartPt;
	Point2f EndPt;
	float lineWidth;

	Point2f Center;
	Point2f unitDir; // [cos(theta), sin(theta)]
	float length;
	float theta;

	// para_a * x + para_b * y + c = 0
	float para_a;
	float para_b;
	float para_c;

	float xMin;
	float xMax;
	float yMin;
	float yMax;
	unsigned short id;
	int colorIdx;
};
/*
 Line line_tmp = LineParameters(startPoint.x startPoint.y, endPoint.x, endPoint.y);
*/
Line LineParameters(float x1, float y1, float x2, float y2, float lineWidth = 1.0f);


#endif //PYLBD_DATACONV_H
