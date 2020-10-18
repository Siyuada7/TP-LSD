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

#include <opencv2/core/persistence.hpp>
#include "../include/LBDFloatLineMatcher.h"
#include "../include/LLDLineMatcher.h"
#include "../include/LinesConfigurator.h"
#include "../include/LBDFloatExtractor.h"
#include "../include/LLDExtractor.h"

#include "../include/EmptyExtractor.h"
#include "../include/EmptyMatcher.h"

StoredLineExtractor* LinesConfigurator::CreateLineExtractor(const std::string &strSettings, bool isLeft)
{
    cv::FileStorage fSettings(strSettings, cv::FileStorage::READ);
    StoredLineExtractor* extractor = NULL;
    bool isTest = (int)fSettings["test"];
    if (fSettings["ldType"] == "LBDFloat")
    {
        std::cout << "Using LBD Float stored, test = " << isTest << std::endl;
        extractor = new LBDFloatExtractor(fSettings["lineDetectionsPath"], fSettings["lineDescriptorsPath"], isLeft, isTest);
    }
    if (fSettings["ldType"] == "LLD")
    {
        std::cout << " Using learned line descriptor " << (std::string) fSettings["lineDetectionsPath"] << std::endl;
        std::cout << "   " << (std::string) fSettings["lineDescriptorsPath"] << std::endl;
        extractor = new LLDExtractor(fSettings["lineDetectionsPath"], fSettings["lineDescriptorsPath"], isLeft, isTest);
    }
    if (!extractor)
    {
        extractor = NULL;
    }
    return extractor;
}

LineMatcher* LinesConfigurator::CreateLineMatcher(const cv::FileStorage& fSettings) {
    LineMatcher* matcher = static_cast<LineMatcher*> (NULL);
    std::cout << " creating LM " << std::endl;
    if (fSettings["ldType"] == "LBDFloat")
    {
        matcher = new LBDFloatLineMatcher();
        std::cout << " LBD LM " << std::endl;
    }
    if (fSettings["ldType"] == "LLD")
    {
        matcher = new LLDLineMatcher();
        std::cout << " LLD LM " << std::endl;
    }
    if (!matcher)
    {
        matcher = new EmptyMatcher();
        std::cout << " Empty LM " << std::endl;
    }
    return matcher;
}