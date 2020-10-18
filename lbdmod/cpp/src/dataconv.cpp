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
#include "../include/dataconv.h"

cv::Mat ConvertKeyLines2Mat(const std::vector<cv::line_descriptor::KeyLine>& detections)
{
    cv::Mat lines_data(detections.size(), 17, CV_32FC1);
    for (int i = 0; i < detections.size(); i++)
    {
        const cv::line_descriptor::KeyLine& kl = detections[i];
        lines_data.at<float>(i, 0) = kl.angle;

        /** object ID, that can be used to cluster keylines by the line they represent */
        lines_data.at<float>(i, 1) = kl.class_id;

        /** octave (pyramid layer), from which the keyline has been extracted */
        lines_data.at<float>(i, 2) = kl.octave;

        /** coordinates of the middlepoint */
        lines_data.at<float>(i, 3) = kl.pt.x;
        lines_data.at<float>(i, 4) = kl.pt.y;

        /** the response, by which the strongest keylines have been selected.
         It's represented by the ratio between line's length and maximum between
         image's width and height */
        lines_data.at<float>(i, 5) = kl.response;

        /** minimum area containing line */
        lines_data.at<float>(i, 6) = kl.size;

        /** lines's extremes in original image */
        lines_data.at<float>(i, 7) = kl.startPointX;
        lines_data.at<float>(i, 8) = kl.startPointY;
        lines_data.at<float>(i, 9) = kl.endPointX;
        lines_data.at<float>(i, 10) = kl.endPointY;

        /** line's extremes in image it was extracted from */
        lines_data.at<float>(i, 11) = kl.sPointInOctaveX;
        lines_data.at<float>(i, 12) = kl.sPointInOctaveY;
        lines_data.at<float>(i, 13) = kl.ePointInOctaveX;
        lines_data.at<float>(i, 14) = kl.ePointInOctaveY;

        /** the length of line */
        lines_data.at<float>(i, 15) = kl.lineLength;

        /** number of pixels covered by the line */
        lines_data.at<float>(i, 16) = kl.numOfPixels;
    }
    return lines_data;
}


std::vector<cv::line_descriptor::KeyLine> ConvertMat2KeyLines(const cv::Mat& lines_data)
{
    std::vector<cv::line_descriptor::KeyLine> detections(lines_data.rows);
    for (int i = 0; i < lines_data.rows; i++)
    {
        cv::line_descriptor::KeyLine kl;
        kl.angle = lines_data.at<float>(i, 0);
        kl.class_id == int(lines_data.at<float>(i, 1));
        kl.octave = int(lines_data.at<float>(i, 2));
        kl.pt.x = lines_data.at<float>(i, 3);
        kl.pt.y = lines_data.at<float>(i, 4);
        kl.response = lines_data.at<float>(i, 5);
        kl.size = lines_data.at<float>(i, 6);
        kl.startPointX = lines_data.at<float>(i, 7);
        kl.startPointY = lines_data.at<float>(i, 8);
        kl.endPointX = lines_data.at<float>(i, 9);
        kl.endPointY = lines_data.at<float>(i, 10);
        kl.sPointInOctaveX = lines_data.at<float>(i, 11);
        kl.sPointInOctaveY = lines_data.at<float>(i, 12);
        kl.ePointInOctaveX = lines_data.at<float>(i, 13);
        kl.ePointInOctaveY = lines_data.at<float>(i, 14);
        kl.lineLength = lines_data.at<float>(i, 15);
        kl.numOfPixels = int(lines_data.at<float>(i, 16));
        detections[i] = kl;
    }
    return detections;
}


cv::Mat KeylineconvertMat(const std::vector<cv::line_descriptor::KeyLine>& detections){
    int rows = detections.size();
    cv::Mat res(rows, 4, CV_32FC1);
    for(int i=0; i<rows; i++){
        res.at<float>(i, 0) = detections[i].startPointX;
        res.at<float>(i, 1) = detections[i].startPointY;
        res.at<float>(i, 2) = detections[i].endPointX;
        res.at<float>(i, 3) = detections[i].endPointY;
    }
    return res;
}

std::vector<cv::line_descriptor::KeyLine> lineconvertKeyline(cv::Mat array, int width, int height){
    std::vector<cv::line_descriptor::KeyLine> keyline;
    int cols = array.cols, rows = array.rows;
    for(int i=0; i<rows; i++){
        Line tmp = LineParameters(array.at<float>(i, 0), array.at<float>(i, 1), array.at<float>(i, 2), array.at<float>(i, 3));
        cv::line_descriptor::KeyLine kl;
		/* fill KeyLine's fields */
		kl.startPointX = tmp.StartPt.x;
		kl.startPointY = tmp.StartPt.y;
		kl.endPointX = tmp.EndPt.x;
		kl.endPointY = tmp.EndPt.y;
		kl.sPointInOctaveX = tmp.StartPt.x;
		kl.sPointInOctaveY = tmp.StartPt.y;
		kl.ePointInOctaveX = tmp.EndPt.x;
		kl.ePointInOctaveY = tmp.EndPt.y;
		kl.lineLength = tmp.length;

		/* compute number of pixels covered by line */
		kl.numOfPixels = (int)tmp.length;

		kl.angle = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
		kl.class_id = i;
		kl.octave = 0;
		kl.size = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
		kl.response = kl.lineLength / max(width , height );
		kl.pt = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );
        keyline.push_back(kl);
    }
    return keyline;
}

Line LineParameters(float x1, float y1, float x2, float y2, float lineWidth)
{
	float dx = x2 - x1;
	float dy = y2 - y1;
	float len = dx * dx + dy * dy;
	len = sqrtf(len);

	Line aLine;
	aLine.StartPt = Point2f(x1, y1);
	aLine.EndPt = Point2f(x2, y2);
	aLine.lineWidth = lineWidth;
	aLine.length = len;
	aLine.theta = atan2(dy, dx);
	aLine.Center = Point2f((x1 + x2) / 2, (y1 + y2) / 2);
	aLine.para_a = -dy;
	aLine.para_b = dx;
	aLine.para_c = x1 * y2 - x2 * y1;
	aLine.unitDir = Point2f(dx / len, dy / len);
	aLine.xMin = min(x1, x2);
	aLine.xMax = max(x1, x2);
	aLine.yMin = min(y1, y2);
	aLine.yMax = max(y1, y2);
	aLine.colorIdx = rand();
	return aLine;
}

cv::Mat ConvertMatches2Mat(const std::vector<cv::DMatch>& matchingVec)
{
    cv::Mat matchingMat = cv::Mat(matchingVec.size(), 4, CV_32SC1);
    for (int i = 0; i < matchingVec.size(); i++)
    {
        const cv::DMatch& dMatch = matchingVec[i];
        matchingMat.at<int>(i, 0) = dMatch.queryIdx;
        matchingMat.at<int>(i, 1) = dMatch.trainIdx;
        matchingMat.at<int>(i, 2) = dMatch.imgIdx;
        matchingMat.at<int>(i, 3) = (int)(dMatch.distance*1000);
    }
    return matchingMat;
}
std::vector<cv::DMatch> ConvertMat2Matches(const cv::Mat& matchMat)
{
    std::vector<cv::DMatch> matchingVec;
    for (int i = 0; i < matchMat.rows; i++)
    {
        cv::DMatch dMatch(matchMat.at<int>(i, 0), matchMat.at<int>(i, 1), 0.001*((float)(matchMat.at<int>(i, 3))));
//        dMatch.queryIdx = matchMat.at<int>(i, 0);
//        dMatch.trainIdx = matchMat.at<int>(i, 1);
//        dMatch.imgIdx = matchMat.at<int>(i, 2);
//        dMatch.distance = 0.001*((float)(matchMat.at<int>(i, 3)));
        matchingVec.push_back(dMatch);
    }
//    std::cout << " filled matching vec " << matchingVec.size() << std::endl;
    return matchingVec;
}