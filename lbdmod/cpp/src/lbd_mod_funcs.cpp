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

#include "../include/lbd_mod_funcs.h"

void DetectEDLines(const cv::Mat& image, int n_octaves, double factor, std::vector<cv::line_descriptor::KeyLine>* detections_p)
{
    cv::line_descriptor::BinaryDescriptor::Params p;
    p.numOfOctave_ = n_octaves;
    p.widthOfBand_ = 7;
    p.ksize_ = 5;
    p.factor = factor;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor(p);

    bd->detect(image, *detections_p, cv::Mat());
}

void DetectComputeLBD(const cv::Mat& image, int n_octaves, double factor, std::vector<cv::line_descriptor::KeyLine>* detections_p, cv::Mat* descs_p)
{
    cv::line_descriptor::BinaryDescriptor::Params p;
    p.numOfOctave_ = n_octaves;
    p.widthOfBand_ = 7;
    p.ksize_ = 5;
    p.factor = factor;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor(p);
    (*bd)(image, cv::Mat(), *detections_p, *descs_p, false, false);
}

void ComputeLBD(const cv::Mat& image, int n_octaves, double factor, std::vector<cv::line_descriptor::KeyLine>& detections_p, cv::Mat* descs_p)
{
    cv::line_descriptor::BinaryDescriptor::Params p;
    p.numOfOctave_ = n_octaves;
    p.widthOfBand_ = 7;
    p.ksize_ = 5;
    p.factor = factor;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor(p);
    bd->compute(image, detections_p, *descs_p, false);
}


void MatchLBD(const cv::Mat& descs1, const cv::Mat& descs2, std::vector<cv::DMatch>* matches_p)
{
    int MATCHES_DIST_THRESHOLD = 25;
    cv::line_descriptor::BinaryDescriptorMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descs1, descs2, matches);
    std::vector<cv::DMatch> good_matches;
	for (int i = 0; i < (int)matches.size(); i++)
	{
		if (matches[i].distance < MATCHES_DIST_THRESHOLD)
			good_matches.push_back(matches[i]);
	}
	*matches_p = good_matches;
}
