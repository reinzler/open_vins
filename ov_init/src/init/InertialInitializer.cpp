/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "InertialInitializer.h"

#ifndef __ANDROID__
#include "dynamic/DynamicInitializer.h"
#endif
#include "static/StaticInitializer.h"

#include "feat/FeatureHelper.h"
#include "types/Type.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"
#include <iostream>
#include <cstdlib>
#include <string>

using namespace ov_core;
using namespace ov_type;
using namespace ov_init;

namespace {
bool glim_openvins_debug_enabled() {
  const char* v = std::getenv("GLIM_OPENVINS_DEBUG");
  if (v == nullptr) {
    return false;
  }
  const std::string value(v);
  return value == "1" || value == "true" || value == "TRUE" || value == "debug" || value == "DEBUG";
}
}  // namespace


InertialInitializer::InertialInitializer(InertialInitializerOptions &params_, std::shared_ptr<ov_core::FeatureDatabase> db)
    : params(params_), _db(db) {

  // Vector of our IMU data
  imu_data = std::make_shared<std::vector<ov_core::ImuData>>();

  // Create initializers
  init_static = std::make_shared<StaticInitializer>(params, _db, imu_data);
#ifndef __ANDROID__
  init_dynamic = std::make_shared<DynamicInitializer>(params, _db, imu_data);
#else
  init_dynamic = nullptr;
#endif
}

void InertialInitializer::feed_imu(const ov_core::ImuData &message, double oldest_time) {

  // Append it to our vector
  imu_data->emplace_back(message);

  // Sort our imu data (handles any out of order measurements)
  // std::sort(imu_data->begin(), imu_data->end(), [](const IMUDATA i, const IMUDATA j) {
  //    return i.timestamp < j.timestamp;
  //});

  // Loop through and delete imu messages that are older than our requested time
  // std::cout << "INIT: imu_data.size() " << imu_data->size() << std::endl;
  if (oldest_time != -1) {
    auto it0 = imu_data->begin();
    while (it0 != imu_data->end()) {
      if (it0->timestamp < oldest_time) {
        it0 = imu_data->erase(it0);
      } else {
        it0++;
      }
    }
  }
}

bool InertialInitializer::initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                                     std::shared_ptr<ov_type::IMU> t_imu, bool wait_for_jerk) {

  static int glim_init_attempt = 0;
  glim_init_attempt++;
  const bool glim_dbg = glim_openvins_debug_enabled() && (glim_init_attempt <= 20 || glim_init_attempt % 50 == 0);

  if (glim_dbg) {
    std::cout << "[openvins_init_core_dbg] enter attempt=" << glim_init_attempt
              << " wait_for_jerk=" << wait_for_jerk
              << " db_features=" << _db->get_internal_data().size()
              << " imu_buffer=" << imu_data->size()
              << " init_window_time=" << params.init_window_time
              << " init_imu_thresh=" << params.init_imu_thresh
              << " init_max_disparity=" << params.init_max_disparity
              << " init_dyn_use=" << params.init_dyn_use
              << " init_dyn_min_deg=" << params.init_dyn_min_deg
              << std::endl;
  }

  // Get the newest and oldest timestamps we will try to initialize between!
  double newest_cam_time = -1;
  for (auto const &feat : _db->get_internal_data()) {
    for (auto const &camtimepair : feat.second->timestamps) {
      for (auto const &time : camtimepair.second) {
        newest_cam_time = std::max(newest_cam_time, time);
      }
    }
  }
  double oldest_time = newest_cam_time - params.init_window_time - 0.10;
  if (newest_cam_time < 0 || oldest_time < 0) {
    if (glim_dbg) {
      std::cout << "[openvins_init_core_dbg] reject attempt=" << glim_init_attempt
                << " reason=no_valid_camera_window"
                << " newest_cam_time=" << newest_cam_time
                << " oldest_time=" << oldest_time
                << std::endl;
    }
    return false;
  }

  if (glim_dbg) {
    std::cout << "[openvins_init_core_dbg] window attempt=" << glim_init_attempt
              << " newest_cam_time=" << newest_cam_time
              << " oldest_time=" << oldest_time
              << std::endl;
  }

  // Remove all measurements that are older then our initialization window
  // Then we will try to use all features that are in the feature database!
  _db->cleanup_measurements(oldest_time);
  auto it_imu = imu_data->begin();
  while (it_imu != imu_data->end() && it_imu->timestamp < oldest_time + params.calib_camimu_dt) {
    it_imu = imu_data->erase(it_imu);
  }

  if (glim_dbg) {
    const double imu_first = imu_data->empty() ? -1.0 : imu_data->front().timestamp;
    const double imu_last = imu_data->empty() ? -1.0 : imu_data->back().timestamp;
    if (glim_openvins_debug_enabled()) {
    std::cout << "[openvins_init_core_dbg] after_cleanup attempt=" << glim_init_attempt
              << " db_features=" << _db->get_internal_data().size()
              << " imu_buffer=" << imu_data->size()
              << " imu_first=" << imu_first
              << " imu_last=" << imu_last
              << " calib_camimu_dt=" << params.calib_camimu_dt
              << std::endl;
    }
  }

  // Compute the disparity of the system at the current timestep
  // If disparity is zero or negative we will always use the static initializer
  bool disparity_detected_moving_1to0 = false;
  bool disparity_detected_moving_2to1 = false;
  if (params.init_max_disparity > 0) {

    // Get the disparity statistics from this image to the previous
    // Only compute the disparity for the oldest half of the initialization period
    double newest_time_allowed = newest_cam_time - 0.5 * params.init_window_time;
    int num_features0 = 0;
    int num_features1 = 0;
    double avg_disp0, avg_disp1;
    double var_disp0, var_disp1;
    FeatureHelper::compute_disparity(_db, avg_disp0, var_disp0, num_features0, newest_time_allowed);
    FeatureHelper::compute_disparity(_db, avg_disp1, var_disp1, num_features1, newest_cam_time, newest_time_allowed);

    // Return if we can't compute the disparity
    int feat_thresh = 15;
    if (num_features0 < feat_thresh || num_features1 < feat_thresh) {
      if (glim_openvins_debug_enabled()) {
      std::cout << "[openvins_init_core_dbg] reject attempt=" << glim_init_attempt
                << " reason=not_enough_feats_for_disparity"
                << " num_features0=" << num_features0
                << " num_features1=" << num_features1
                << " feat_thresh=" << feat_thresh
                << std::endl;
      }
      if (glim_openvins_debug_enabled()) {
        PRINT_WARNING(YELLOW "[init]: not enough feats to compute disp: %d,%d < %d\n" RESET, num_features0, num_features1, feat_thresh);
      }
      return false;
    }

    // Check if it passed our check!
    if (glim_openvins_debug_enabled()) {
    std::cout << "[openvins_init_core_dbg] disparity attempt=" << glim_init_attempt
              << " avg0=" << avg_disp0
              << " avg1=" << avg_disp1
              << " var0=" << var_disp0
              << " var1=" << var_disp1
              << " num_features0=" << num_features0
              << " num_features1=" << num_features1
              << " thresh=" << params.init_max_disparity
              << std::endl;
    }
    if (glim_openvins_debug_enabled()) {
      PRINT_INFO(YELLOW "[init]: disparity is %.3f,%.3f (%.2f thresh)\n" RESET, avg_disp0, avg_disp1, params.init_max_disparity);
    }
    disparity_detected_moving_1to0 = (avg_disp0 > params.init_max_disparity);
    disparity_detected_moving_2to1 = (avg_disp1 > params.init_max_disparity);
  }

  // Use our static initializer!
  // CASE1: if our disparity says we were static in last window and have moved in the newest, we have a jerk
  // CASE2: if both disparities are below the threshold, then the platform has been stationary during both periods
  bool has_jerk = (!disparity_detected_moving_1to0 && disparity_detected_moving_2to1);
  bool is_still = (!disparity_detected_moving_1to0 && !disparity_detected_moving_2to1);

  if (glim_dbg) {
    std::cout << "[openvins_init_core_dbg] mode_decision attempt=" << glim_init_attempt
              << " moving_1to0=" << disparity_detected_moving_1to0
              << " moving_2to1=" << disparity_detected_moving_2to1
              << " has_jerk=" << has_jerk
              << " is_still=" << is_still
              << " wait_for_jerk=" << wait_for_jerk
              << " init_imu_thresh=" << params.init_imu_thresh
              << " init_dyn_use=" << params.init_dyn_use
              << std::endl;
  }

  if (((has_jerk && wait_for_jerk) || (is_still && !wait_for_jerk)) && params.init_imu_thresh > 0.0) {
    PRINT_DEBUG(GREEN "[init]: USING STATIC INITIALIZER METHOD!\n" RESET);

    if (glim_openvins_debug_enabled()) {
      std::cout << "[openvins_init_core_dbg] using_static attempt="
                << glim_init_attempt
                << std::endl;
    }

    bool static_success = init_static->initialize(timestamp, covariance, order, t_imu, wait_for_jerk);

    if (glim_openvins_debug_enabled()) {
      std::cout << "[openvins_init_core_dbg] static_result attempt="
                << glim_init_attempt
                << " success=" << static_success
                << " timestamp=" << timestamp
                << " cov=" << covariance.rows() << "x" << covariance.cols()
                << " order_size=" << order.size()
                << std::endl;
    }

    return static_success;
  } else if (params.init_dyn_use && !is_still) {
#ifndef __ANDROID__
    if (glim_openvins_debug_enabled()) {
    std::cout << "[openvins_init_core_dbg] using_dynamic attempt=" << glim_init_attempt << std::endl;
    PRINT_DEBUG(GREEN "[init]: USING DYNAMIC INITIALIZER METHOD!\n" RESET);
    }
    std::map<double, std::shared_ptr<ov_type::PoseJPL>> _clones_IMU;
    std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> _features_SLAM;
    if (init_dynamic) {
      return init_dynamic->initialize(timestamp, covariance, order, t_imu, _clones_IMU, _features_SLAM);
    }
#else
    PRINT_ERROR(RED "[init]: DYNAMIC INITIALIZER not available on Android (Ceres Solver not included)\n" RESET);
#endif
  } else {
    std::string msg = (has_jerk) ? "" : "no accel jerk detected";
    msg += (has_jerk || is_still) ? "" : ", ";
    msg += (is_still) ? "" : "platform moving too much";
    PRINT_INFO(YELLOW "[init]: failed static init: %s\n" RESET, msg.c_str());
  }
  return false;
}
