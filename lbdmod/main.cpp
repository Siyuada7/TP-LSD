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

#include <iostream>
#include "cpp/src/precomp.hpp"
#include "cpp/include/lbd_mod_funcs.h"

int main()
{

    bool is_simultaneous = false;
    bool is_highgui = false;

    cv::Mat image = cv::imread("test_imgs/000000.png", 0);
    cv::Mat image2 = cv::imread("test_imgs/000001.png", 0);
    cv::line_descriptor::BinaryDescriptor::Params p;
    int n_octaves = 8;
    p.numOfOctave_ = n_octaves;
    p.widthOfBand_ = 7;
    p.ksize_ = 5;
    p.factor = 1.2;

    std::cout << " params created " << std::endl;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor(p);
    cv::Mat mask;
    long long t0 = cv::getTickCount();
    std::cout << " detect start " << std::endl;
    cv::Mat descs;
    std::vector<cv::line_descriptor::KeyLine> lines;
    if (is_simultaneous)
    {
        (*bd)(image, cv::Mat(), lines, descs, false, false);
    } else {

        DetectEDLines(image, n_octaves, 1.2, &lines);
        ComputeLBD(image, n_octaves, 1.2, lines, &descs);
    }


    long long t1 = cv::getTickCount();
    double dt = (t1 - t0) / cv::getTickFrequency();

    cv::Mat debugDetector, imageColor;
    cv::cvtColor(image, imageColor, CV_GRAY2BGR);
    cv::line_descriptor::drawKeylines(imageColor, lines, debugDetector);

    if (is_highgui)
        cv::imshow("detect 1", debugDetector);

    cv::Mat descs2;
    std::vector<cv::line_descriptor::KeyLine> lines2;
    if (is_simultaneous) {
        (*bd)(image2, cv::Mat(), lines2, descs2, false, false);
    } else {
//        DetectComputeLBD(image2, n_octaves, 1.2, &lines2, &descs2);
        DetectEDLines(image, n_octaves, 1.2, &lines2);
        ComputeLBD(image, n_octaves, 1.2, lines2, &descs2);
    }

    cv::Mat debugDetector2, imageColor2;
    cv::cvtColor(image2, imageColor2, CV_GRAY2BGR);
    cv::line_descriptor::drawKeylines(imageColor2, lines2, debugDetector2);

    if (is_highgui)
        cv::imshow("detect 2", debugDetector2);


    cv::line_descriptor::BinaryDescriptorMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descs2, descs, matches);
    cv::Mat debugMatchingImg;
    bool is_vertical = true;
    cv::line_descriptor::drawLineMatches(imageColor2, lines2, imageColor, lines, matches, debugMatchingImg, true);
    cv::imwrite("test.png", debugMatchingImg);

    if (is_highgui)
    {
        cv::imshow("Matching", debugMatchingImg);
        cv::waitKey();
    }

    return 0;
}
