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

#include <fstream>
#include "../include/StoredLineExtractor.h"

StoredLineExtractor::StoredLineExtractor(const std::string &strDetectionsStorage, bool isLeft, bool isTest) :
        strDetectionsStorage(strDetectionsStorage), isLeft(isLeft), frameId(0), isTest(isTest)
{}

int FindPoseInBatch(int currFrameId, bool isLeft, int frameId, int maxTempDist)
{
    int cnt = 0;
    int posInBatch = -1;

    for (int i = currFrameId+maxTempDist; i < currFrameId+maxTempDist+2; i++)
    {
        if (i == frameId)
        {
            if (isLeft)
            {
                posInBatch = 2*cnt;
            } else {
                posInBatch = 2*cnt+1;
            }
        }
        cnt++;
    }

    for (int j = 1; j < maxTempDist+1; j++)
    {
        for (int k = -1; k < 2; k+=2)
        {
            int i = currFrameId+maxTempDist+k*j;
            if (i == currFrameId + maxTempDist || i == currFrameId + maxTempDist+1)
            {
                continue;
            }
            if (i == frameId) {
                if (isLeft) {
                    posInBatch = 2 * cnt;
                } else {
                    posInBatch = 2 * cnt + 1;
                }
            }
            cnt++;
        }
    }
    return posInBatch;
}

void StoredLineExtractor::LoadDetectionsTrainDataset(int frameId)
{
    int maxTempDist = 5;
    int currFrameId = frameId/maxTempDist;
    currFrameId *= maxTempDist;
    posInBatch = FindPoseInBatch(currFrameId, isLeft, frameId, maxTempDist);
    batchId = currFrameId;
    std::string linePath = strDetectionsStorage + "/" + std::to_string(batchId) + "/" + std::to_string(posInBatch)+"_l.png";
//    std::cout << " reading dets from " << linePath << std::endl;
    linesMat = cv::imread(linePath, CV_LOAD_IMAGE_UNCHANGED);

    if (linesMat.empty())
    {
        posInBatch = FindPoseInBatch(currFrameId-maxTempDist, isLeft, frameId, maxTempDist);
        batchId = currFrameId-maxTempDist;
        linePath = strDetectionsStorage + "/" + std::to_string(batchId) + "/" + std::to_string(posInBatch)+"_l.png";
//        std::cout << " reading dets from " << linePath << std::endl;
        linesMat = cv::imread(linePath, CV_LOAD_IMAGE_UNCHANGED);

        if (linesMat.empty())
        {
            posInBatch = FindPoseInBatch(currFrameId+maxTempDist, isLeft, frameId, maxTempDist);
            batchId = currFrameId+maxTempDist;
            linePath = strDetectionsStorage + "/" + std::to_string(batchId) + "/" + std::to_string(posInBatch)+"_l.png";
//            std::cout << " reading dets from " << linePath << std::endl;
            linesMat = cv::imread(linePath, CV_LOAD_IMAGE_UNCHANGED);
        }
    }
}

void StoredLineExtractor::LoadDetectionsTestDataset(int frameId)
{
    std::string suff;
    if (isLeft)
    {
        suff = "_0";
    } else {
        suff = "_1";
    }
    std::string linePath = strDetectionsStorage + "/" + std::to_string(frameId) + suff + ".png";
    std::cout << " reading dets from " << linePath << std::endl;
    linesMat = cv::imread(linePath, CV_LOAD_IMAGE_UNCHANGED);
//    std::cout << "dets read " << linesMat.cols << std::endl;
}

void StoredLineExtractor::SetFrameId(int frameId)
{
    this->frameId = frameId;
    if (isTest)
    {
        LoadDetectionsTestDataset(frameId);
    } else {
        LoadDetectionsTrainDataset(frameId);
    }

}

void StoredLineExtractor::ExtractDetections(const cv::Mat& frame, std::vector<KeyLine> *extrLines)
{
//    std::string linePath = strDetectionsStorage + "/" + std::to_string(batchId) + "/" + std::to_string(posInBatch)+"_l.png";
//    std::cout << " reading dets from " << linePath << std::endl;
//    cv::Mat linesMat = cv::imread(linePath, CV_LOAD_IMAGE_UNCHANGED);
    extrLines->clear();
    for (int li = 0; li < linesMat.cols; li++)
    {
        std::vector<KeyLine> line_dets;
        KeyLine kl;
        kl.startPointX = linesMat.at<ushort>(0, li);
        kl.startPointY = linesMat.at<ushort>(1, li);
        kl.endPointX = linesMat.at<ushort>(2, li);
        kl.endPointY = linesMat.at<ushort>(3, li);
        kl.octave = linesMat.at<ushort>(4, li);
//        std::cout << "linedet: " << kl.startPointX << " " << kl.startPointY << " " << kl.endPointX << " " << kl.endPointY << " " << kl.octave << std::endl;
        kl.angle = 0;
        kl.response = 1;
        kl.numOfPixels = 0;
        kl.size = 0;
        extrLines->push_back(kl);
    }
}

void StoredLineExtractor::ReadTXTDescriptors(std::vector<KeyLine> &extrLines, const std::string& strDescriptorsStorage,
                                             cv::Mat *lineDescs)
{
    std::string curr_descs_path = strDescriptorsStorage + std::to_string(frameId)+"_0.txt";
    if (!isLeft)
    {
        curr_descs_path = strDescriptorsStorage + std::to_string(frameId)+"_1.txt";
    }
//    std::cout << " reading descs from " << curr_descs_path << std::endl;
    std::ifstream desc_reader(curr_descs_path);
    std::string line;
    std::vector<cv::Mat> descs;
    descs.clear();
    while (std::getline(desc_reader, line))
    {
        std::istringstream iss(line);
        std::vector<double> data_row;
        double v;
        while (iss >> v) {
            data_row.push_back(v);
        }
        cv::Mat data(1, data_row.size(), CV_64FC1);
        for (int i = 0; i < data_row.size(); i++)
        {
            data.at<double>(0, i) = data_row[i];
        }
        data = data / cv::norm(data);
        descs.push_back(data);
    }
    if (descs.size() > 0)
    {
        cv::Mat lds = cv::Mat(descs.size(), descs[0].cols, CV_64FC1);
        for (int i = 0; i < descs.size(); i++)
        {
            descs[i].copyTo(lds.row(i));
        }
        *lineDescs = lds.clone();
    }

    std::cout << " read " << descs.size() << " line descs " << std::endl;

    if (descs.size() != extrLines.size())
    {
        for (int i = 0; i < 10; i++)
        {
            std::cout << " ERR reading descs " << descs.size() << " " << extrLines.size() << " " << frameId << std::endl;

            std::cout << curr_descs_path << std::endl;
        }
        extrLines.clear();
        *lineDescs = cv::Mat();
    }
}