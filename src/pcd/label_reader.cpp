// Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2016/12/22 16:36:23
// @file label_reader.cpp
// @brief 
// 
#include "pcd/label_reader.h"

#include <exception>

namespace adu {
namespace perception {

bool LabelsReader::init(const std::string& file_name) {
    _labels_file_name = file_name;
    //Read Data from line 
    std::ifstream ifs;
    ifs.open(_labels_file_name.c_str(), std::fstream::in);
    if (!ifs) {
        std::cerr << "Open File:" << _labels_file_name << " failed. e:" << strerror(errno) << std::endl;
        return false;
    }
    //Parse Data from Columns
    std::string line;
    while (std::getline(ifs, line)) {

        std::vector<std::string> columns;
        boost::split(columns, line, boost::is_any_of(" \t"));
        const std::string& pcd_file_name = columns[2].substr(8); //"./files/xxx" -> "xxx"
        std::string& json_data = columns[3];

        std::stringstream json_stream;
        json_stream << json_data;

        pt::ptree root;
        pt::read_json(json_stream, root);
        _labels[pcd_file_name] = Label::Ptr(new Label(pcd_file_name, root));
//        if (pcd_file_name == "QB9178_12_1461753402_1461753702_3641.pcd") {
//            std::cout << _labels[pcd_file_name]->debug_string();
//        }
//        std::cout << "LabelsReader pcd_file_name:" << pcd_file_name << std::endl;
    }
    return true;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr read_pcd(const std::string& pcd_file_name) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>); 
    //Point Cloud MonochromeCloud(pcl::PointCloud<pcl::PointXYZ>)
    int offset = 0; //[input] offset, you may change this when read from tar file.

    //Read Point Cloud from File    
    pcl::PCDReader file_reader;
    int ret = file_reader.read(pcd_file_name, *point_cloud, offset);
    if (ret != 0) {
        LOG(ERROR) << "file:" << pcd_file_name << " open failed. ret:" << ret << std::endl;
        return pcl::PointCloud<pcl::PointXYZ>::Ptr();
    }
    return point_cloud;
}    


} // namespace perception
} // namespace adu
