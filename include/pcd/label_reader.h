// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/01/06 10:58:13
// @file include/pcd/label_reader.h
// @brief 
#ifndef INCLUDE_PCD_LABEL_READER_H
#define INCLUDE_PCD_LABEL_READER_H

#include <pcl/common/common_headers.h>
#include <pcl/range_image/range_image.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include "pcd/types.h"
#include "glog/logging.h"

namespace adu {
namespace perception {

class LabelsReader {
public:
    typedef boost::unordered_map<std::string, Label::Ptr>::iterator Iter;
public:
    bool init(const std::string& file_name);
    
    const Label::Ptr& get(const std::string& pcd_file_name) const {
//TODO: ADD mutex.
        return _labels[pcd_file_name];
    }
    const Iter begin() {
        return _labels.begin();
    }
    const Iter end() {
        return _labels.end();
    }
private:
    std::string _labels_file_name;
    mutable boost::unordered_map<std::string, Label::Ptr> _labels;
};

pcl::PointCloud<pcl::PointXYZ>::Ptr read_pcd(const std::string& pcd_file_name); 
} // namespace perception
} // namespace adu


#endif  // INCLUDE_PCD_LABEL_READER_H
// 

