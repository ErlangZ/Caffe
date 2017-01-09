// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/01/06 14:28:54
// @file include/pcd/grid.h
// @brief 
#ifndef INCLUDE_PCD_GRID_H
#define INCLUDE_PCD_GRID_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <glog/logging.h>

#include "pcd/bounding_box_feature.h"

namespace adu {
namespace perception {

const float resolution = 0.4;

template<typename Dtype>
class Grid {
public:
    Grid(int x_num, int y_num, int z_num, Dtype* data): 
        _x_num(x_num), _y_num(y_num), _z_num(z_num), _data(data) {
        memset(_data, 0, _x_num * _y_num * _z_num);
        for (int i = 0; i < x_num * y_num * z_num; i++) {
            _data[i] = 5.0;
        }
    }

    bool put_point_cloud_to_grids(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
        Eigen::Vector3f min; //0-Theta, 1-Radius, 2-Z
        Eigen::Vector3f max; //0-Theta, 1-Radius, 2-Z
        BoundingBoxFeature bounding_box_features;
        bounding_box_features.min_max(point_cloud, min, max);
 
        Eigen::Vector3f size = max - min;
        if (size(0) > _x_num * resolution) {
            LOG(ERROR) << "Point Cloud Too Big in x-size:" << size(0) 
                        << " X_num:" << _x_num << " Resolution:" << resolution;
            return false;
        }
        if (size(1) > _y_num * resolution) {
            LOG(ERROR) << "Point Cloud Too Big in y-size:" << size(1) 
                        << " Y_num:" << _y_num << " Resolution:" << resolution;
            return false;
        }
        if (size(2) > _z_num * resolution) {
            LOG(ERROR) << "Point Cloud Too Big in z-size:" << size(2) 
                        << " Z_num:" << _z_num << " Resolution:" << resolution;
            return false;
        }
 
        for (int i = 0; i < point_cloud->points.size(); i++) {
            const pcl::PointXYZ& point = point_cloud->points[i];
            int row = (point.x - min(0)) / resolution;
            int col = (point.y - min(1)) / resolution;
            int height = (point.z - min(2)) / resolution;
            _data[row * _y_num * _z_num + col * _z_num + height ] += 20.0;
        } 
        return true;
    }

private:
    int _x_num;
    int _y_num;
    int _z_num;
    Dtype* _data;
};

}
}

#endif  // INCLUDE_PCD_GRID_H
// 

