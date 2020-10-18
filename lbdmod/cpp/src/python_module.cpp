#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include "pyboostcvconverter/pyboostcvconverter.hpp"
#include "../include/lbd_mod_funcs.h"
#include "../include/dataconv.h"

namespace pylbd {

    using namespace boost::python;

    cv::Mat detect_edlines(cv::Mat image, int n_octaves, double factor)
    {
        std::vector<cv::line_descriptor::KeyLine> detections;
        DetectEDLines(image, n_octaves, factor, &detections);
        cv::Mat lines_data = KeylineconvertMat(detections);

        return lines_data;
    }

    boost::python::tuple detect_and_describe(cv::Mat image, int n_octaves, double factor)
    {
        std::vector<cv::line_descriptor::KeyLine> detections;
        cv::Mat descs;
        DetectComputeLBD(image, n_octaves, factor, &detections, &descs);
        cv::Mat lines_data = ConvertKeyLines2Mat(detections);
        return boost::python::make_tuple(descs, lines_data);
    }

    cv::Mat detect_lsd(cv::Mat image, int n_octaves, double factor)
    {
        std::vector<cv::line_descriptor::KeyLine> detections;
        cv::Ptr<cv::line_descriptor::LSDDetector> lsd_detector;
        lsd_detector = cv::line_descriptor::LSDDetector::createLSDDetector();
        lsd_detector->detect(image, detections, 2, 2);
        cv::Mat lines_data = KeylineconvertMat(detections);

        return lines_data;
    }

    boost::python::tuple detect_and_describe_lsd(cv::Mat image, int n_octaves, double factor)
    {
        std::vector<cv::line_descriptor::KeyLine> detections;
        cv::Ptr<cv::line_descriptor::LSDDetector> lsd_detector;
        lsd_detector = cv::line_descriptor::LSDDetector::createLSDDetector();
        lsd_detector->detect(image, detections, 2, 2);
        cv::Mat descs;
        cv::Ptr<cv::line_descriptor::BinaryDescriptor> bd_ = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        bd_->compute(image, detections, descs);

        cv::Mat lines_data = ConvertKeyLines2Mat(detections);
        return boost::python::make_tuple(descs, lines_data);
    }

    cv::Mat describe_with_lbd(cv::Mat image, cv::Mat lines1, int n_octaves, double factor)
    {
        std::vector<cv::line_descriptor::KeyLine> detections = lineconvertKeyline(lines1);
        cv::Mat descs;
        ComputeLBD(image, n_octaves, factor, detections, &descs);
        return descs;
    }

    cv::Mat match_lbd_descriptors(cv::Mat desc1, cv::Mat desc2)
    {
        std::vector<cv::DMatch> matchingVec;
        MatchLBD(desc1, desc2, &matchingVec);
        cv::Mat matchingMat = ConvertMatches2Mat(matchingVec);
        return matchingMat;
    }

    cv::Mat visualize_line_matching(cv::Mat image1, cv::Mat lines1, cv::Mat image2, cv::Mat lines2, cv::Mat matches, bool is_vertical)
    {
        std::vector<cv::line_descriptor::KeyLine> keyLines1 = ConvertMat2KeyLines(lines1);
        std::vector<cv::line_descriptor::KeyLine> keyLines2 = ConvertMat2KeyLines(lines2);
        std::vector<cv::DMatch> matchesVec = ConvertMat2Matches(matches);
        cv::Mat debugMatchingImg;
        cv::line_descriptor::drawLineMatches(image1, keyLines1, image2, keyLines2, matchesVec, debugMatchingImg, is_vertical);
        return debugMatchingImg;
    }


#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pylbd) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
        pbcvt::matFromNDArrayBoostConverter();

        //expose module-level functions
        def("detect_lsd", detect_lsd);
        def("detect_and_describe_lsd", detect_and_describe_lsd);
        def("detect_edlines", detect_edlines);
        def("detect_and_describe", detect_and_describe);
        def("describe_with_lbd", describe_with_lbd);
        def("match_lbd_descriptors", match_lbd_descriptors);
        def("visualize_line_matching", visualize_line_matching);

		//from PEP8 (https://www.python.org/dev/peps/pep-0008/?#prescriptive-naming-conventions)
        //"Function names should be lowercase, with words separated by underscores as necessary to improve readability."
//        def("increment_elements_by_one", increment_elements_by_one);
    }

} //end namespace pylbd
