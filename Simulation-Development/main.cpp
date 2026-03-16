/*
 * main.cpp  —  RobotDrive node setup
 *
 * Initialises:
 *   • TF broadcaster          (world → base_link)
 *   • Odometry publisher      (/odom)
 *   • IMU publisher           (/imu/data)
 *   • cmd_vel subscriber      (/cmd_vel)
 *   • 50 ms physics timer
 *   • Obstacle world          (build_world)
 *   • PID controllers         (constructed in header with tuned gains)
 */

#include "robot_sim/robot_drive.hpp"
#include <tf2/LinearMath/Quaternion.h>

// ─────────────────────────────────────────────────────────────────────────────
RobotDrive::RobotDrive()
    : Node("robot_drive"),
      t(0.0),
      pid_v_    ( 8.0,  1.5, 0.20, 10.0, 30.0 ),
      pid_omega_( 5.0,  0.8, 0.12,  4.0, 10.0 )
{
    // TF broadcaster
    broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // Publishers
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom",     10);
    imu_pub_  = create_publisher<sensor_msgs::msg::Imu>  ("/imu/data", 10);

    // cmd_vel subscriber
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        std::bind(&RobotDrive::cmd_vel_callback, this, std::placeholders::_1));

    // Build the obstacle world
    build_world();

    // 50 ms physics timer
    timer_ = create_wall_timer(
        std::chrono::milliseconds(50),
        std::bind(&RobotDrive::update_pose, this));

    RCLCPP_INFO(get_logger(),
        "RobotDrive v2 started  [PID + Collisions + Slope terrain]");
}

// ─────────────────────────────────────────────────────────────────────────────
void RobotDrive::setPose(double x, double y, double yaw)
{
    // NOTE: update_pose() now broadcasts the TF directly (with roll/pitch for
    // terrain tilt), so this function is kept for compatibility but not
    // called directly from update_pose() any more.
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp    = get_clock()->now();
    tf.header.frame_id = "world";
    tf.child_frame_id  = "base_link";

    tf.transform.translation.x = x;
    tf.transform.translation.y = y;
    tf.transform.translation.z = state_.terrain_z;

    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();

    broadcaster_->sendTransform(tf);
}

// ─────────────────────────────────────────────────────────────────────────────
void RobotDrive::cmd_vel_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg)
{
    cmd_.v_target     = msg->linear.x;
    cmd_.omega_target = msg->angular.z;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotDrive>());
    rclcpp::shutdown();
    return 0;
}
