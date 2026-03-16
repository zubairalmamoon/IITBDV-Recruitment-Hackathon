/*
 * trajectory.cpp  —  Physics Simulation Engine  (v2 — extended)
 * ==============================================================
 *
 * Builds on the base submission and adds three new physics features:
 *
 *  ┌──────────────────────────────────────────────────────────────────────┐
 *  │  FEATURE 1 — PID SPEED CONTROLLER                                   │
 *  │    Two independent PID loops replace the raw wheel-torque mapping:  │
 *  │      • pid_v_     tracks linear  speed (cmd_vel.linear.x)           │
 *  │      • pid_omega_ tracks angular speed (cmd_vel.angular.z)          │
 *  │    Both include anti-windup integral clamping and output saturation. │
 *  │    Ref: https://en.wikipedia.org/wiki/PID_controller                │
 *  ├──────────────────────────────────────────────────────────────────────┤
 *  │  FEATURE 2 — SLOPE / TERRAIN                                        │
 *  │    An analytic sinusoidal heightmap  h(x,y) = A·sin(kx·x)·sin(ky·y)│
 *  │    gives a 3-D terrain. The gradient at the robot position gives the │
 *  │    slope angles. Gravity is decomposed along the slope surface and   │
 *  │    projected into the robot body frame as driving/lateral forces.    │
 *  │    Ref: https://en.wikipedia.org/wiki/Rigid_body_dynamics            │
 *  ├──────────────────────────────────────────────────────────────────────┤
 *  │  FEATURE 3 — AABB COLLISION DETECTION & RESPONSE                   │
 *  │    Static box obstacles are defined in build_world(). Each tick:     │
 *  │      1. Circle-vs-AABB overlap test (nearest-point method).         │
 *  │      2. Positional correction pushes robot out of penetration.      │
 *  │      3. Normal velocity reflected with coefficient of restitution.  │
 *  │      4. Tangential velocity damped by contact friction.             │
 *  └──────────────────────────────────────────────────────────────────────┘
 *
 * Physics pipeline each tick (50 ms):
 *   1.  PID step          → corrective Fx and Tz in body frame
 *   2.  Motor lag filter  → wheel speeds track desired values smoothly
 *   3.  Aero drag         → quadratic air resistance opposes motion
 *   4.  Lateral friction  → non-holonomic side-force constraint
 *   5.  Rolling resistance→ small forward-opposing force
 *   6.  Terrain sampling  → evaluate h(x,y) and gradient
 *   7.  Slope gravity     → gravity component along slope → body forces
 *   8.  Net F = ma        → accelerations from total forces & torques
 *   9.  Euler integration → velocity and position update
 *  10.  Collision resolve → positional push-out + impulse response
 *  11.  Odometry update   → dead-reckoning accumulator
 *
 * References
 * ──────────
 *  PID controller     : https://en.wikipedia.org/wiki/PID_controller
 *  Rigid body dynamics: https://en.wikipedia.org/wiki/Rigid_body_dynamics
 *  Euler method       : https://en.wikipedia.org/wiki/Euler_method
 *  Diff-drive model   : http://rossum.sourceforge.net/papers/DiffSteer/
 *  ROS 2 TF2          : https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html
 */

#include "robot_sim/robot_drive.hpp"
#include <cmath>
#include <random>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2/LinearMath/Quaternion.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers (anonymous namespace — not visible outside this TU)
// ─────────────────────────────────────────────────────────────────────────────
namespace
{

double sign(double v)
{
    if (v >  1e-9) return  1.0;
    if (v < -1e-9) return -1.0;
    return 0.0;
}

double gaussian_noise(double sigma)
{
    static std::mt19937 rng{42};
    static std::normal_distribution<double> dist(0.0, 1.0);
    return sigma * dist(rng);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
//  Constructor  —  initialise ROS infrastructure + world
// ─────────────────────────────────────────────────────────────────────────────
// (defined in main.cpp — only world + PID reset here)
// build_world() is called from main.cpp constructor via RobotDrive::build_world()

void RobotDrive::build_world()
{
    /*
     * Populate the obstacle list.
     *
     * BoxObstacle { cx, cy, half_width_x, half_depth_y }
     *
     * Layout (top-down view, 1 unit = 1 m):
     *
     *   ┌──────────────────────────────────┐   ← North wall
     *   │                                  │
     *   │       [pillar]   [pillar]         │
     *   │                                  │
     *   │  [box]                [box]       │
     *   │                                  │
     *   │       [pillar]   [pillar]         │
     *   │                                  │
     *   └──────────────────────────────────┘   ← South wall
     *
     * The robot starts at (0,0) in the centre of the arena.
     */

    // Boundary walls (thick: 0.1 m)
    obstacles_.push_back({ 0.0,  5.2,  6.0, 0.1 });   // North wall
    obstacles_.push_back({ 0.0, -5.2,  6.0, 0.1 });   // South wall
    obstacles_.push_back({ 5.2,  0.0,  0.1, 5.0 });   // East wall
    obstacles_.push_back({-5.2,  0.0,  0.1, 5.0 });   // West wall

    // Interior pillars (square cross-section 0.25 m)
    obstacles_.push_back({ 2.5,  2.5,  0.25, 0.25 });
    obstacles_.push_back({-2.5,  2.5,  0.25, 0.25 });
    obstacles_.push_back({ 2.5, -2.5,  0.25, 0.25 });
    obstacles_.push_back({-2.5, -2.5,  0.25, 0.25 });

    // Rectangular crates
    obstacles_.push_back({ 1.0,  0.0,  0.30, 0.20 });
    obstacles_.push_back({-1.0,  0.0,  0.30, 0.20 });

    RCLCPP_INFO(get_logger(),
        "World built: %zu obstacles.", obstacles_.size());
}

// ─────────────────────────────────────────────────────────────────────────────
//  update_pose  —  50 ms timer callback
// ─────────────────────────────────────────────────────────────────────────────
void RobotDrive::update_pose()
{
    t += DT;

    physics_step(DT);

    // Pitch and roll from terrain slope for the TF broadcast
    // (RViz can show the robot tilting on slopes)
    double pitch = -std::atan(state_.slope_x);   // tilt about Y-axis
    double roll  =  std::atan(state_.slope_y);   // tilt about X-axis

    tf2::Quaternion q;
    q.setRPY(roll, pitch, state_.yaw);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp    = get_clock()->now();
    tf.header.frame_id = "world";
    tf.child_frame_id  = "base_link";
    tf.transform.translation.x = state_.x;
    tf.transform.translation.y = state_.y;
    tf.transform.translation.z = state_.terrain_z;   // ride terrain height
    tf.transform.rotation.x    = q.x();
    tf.transform.rotation.y    = q.y();
    tf.transform.rotation.z    = q.z();
    tf.transform.rotation.w    = q.w();

    broadcaster_->sendTransform(tf);

    publish_odometry();
    publish_imu();
}

// ─────────────────────────────────────────────────────────────────────────────
//  physics_step  —  one complete simulation tick
// ─────────────────────────────────────────────────────────────────────────────
void RobotDrive::physics_step(double dt)
{
    const double L = params_.wheel_base;
    const double R = params_.wheel_radius;
    const double m = params_.mass;
    const double g = 9.81;
    const double N = m * g;   // Normal force on flat ground

    // ── STEP 1: PID controllers ───────────────────────────────────────────
    //
    //  The PIDs compute corrective forces/torques that close the gap between
    //  the commanded speed and the robot's current speed.
    //
    double Fx_pid = 0.0, Tz_pid = 0.0;
    pid_step(dt, Fx_pid, Tz_pid);

    // ── STEP 2: Motor lag filter ──────────────────────────────────────────
    //
    //  The PID output is a desired force/torque, not a direct wheel command.
    //  We convert back to desired wheel speeds and apply the motor lag:
    //
    //    v_desired   = Fx_pid / m  +  current_vx   (incremental)
    //    om_desired  = Tz_pid / Iz +  current_omega
    //
    double v_des  = state_.vx    + Fx_pid / m  * dt;
    double om_des = state_.omega + Tz_pid / params_.inertia * dt;

    v_des  = std::clamp(v_des,  -params_.max_speed, params_.max_speed);
    om_des = std::clamp(om_des, -params_.max_omega,  params_.max_omega);

    double omW_L_des = (v_des - om_des * L / 2.0) / R;
    double omW_R_des = (v_des + om_des * L / 2.0) / R;

    // First-order motor lag  (τ = 0.15 s)
    const double tau   = 0.15;
    const double alpha = dt / tau;
    state_.omega_left  += alpha * (omW_L_des - state_.omega_left);
    state_.omega_right += alpha * (omW_R_des - state_.omega_right);

    // Actual traction forces from wheel speeds
    double F_L = std::clamp(state_.omega_left  * m * R,
                            -params_.max_wheel_torque / R,
                             params_.max_wheel_torque / R);
    double F_R = std::clamp(state_.omega_right * m * R,
                            -params_.max_wheel_torque / R,
                             params_.max_wheel_torque / R);

    double Fx_drive = F_L + F_R;
    double Tz_drive = (F_R - F_L) * (L / 2.0);

    // ── STEP 3: Aerodynamic drag ──────────────────────────────────────────
    double speed   = std::hypot(state_.vx, state_.vy);
    double F_drag  = 0.5 * params_.rho_air * params_.Cd
                         * params_.frontal_area * speed * speed;
    double Fx_drag = (speed > 1e-6) ? -F_drag * (state_.vx / speed) : 0.0;

    // ── STEP 4: Non-holonomic lateral friction ────────────────────────────
    double F_lat_max = params_.mu_kinetic * N;
    double F_lat     = std::clamp(-m * state_.vy / dt,
                                  -F_lat_max, F_lat_max);

    // ── STEP 5: Rolling resistance ────────────────────────────────────────
    double F_roll = -params_.mu_roll * N * sign(state_.vx);

    // ── STEP 6: Terrain sampling ──────────────────────────────────────────
    //
    //  Heightmap  h(x,y) = A · sin(kx·x) · sin(ky·y)
    //  Gradient:
    //    ∂h/∂x = A · kx · cos(kx·x) · sin(ky·y)
    //    ∂h/∂y = A · ky · sin(kx·x) · cos(ky·y)
    //
    sample_terrain();

    // ── STEP 7: Slope gravity ─────────────────────────────────────────────
    //
    //  On an inclined surface with gradient (slope_x, slope_y) the gravity
    //  vector has a component along the slope:
    //
    //    F_gravity_world_x = −m·g·slope_x   (downhill in world X)
    //    F_gravity_world_y = −m·g·slope_y   (downhill in world Y)
    //
    //  We then project these world-frame forces into the robot body frame
    //  using the robot's heading yaw:
    //
    //    Fx_slope =  Fwx·cos(yaw) + Fwy·sin(yaw)   (forward component)
    //    Fy_slope = -Fwx·sin(yaw) + Fwy·cos(yaw)   (lateral component)
    //
    double Fx_slope = 0.0, Fy_slope = 0.0;
    apply_slope_gravity(Fx_slope, Fy_slope);

    // ── STEP 8: Net forces & Newton's 2nd law ─────────────────────────────
    double Fx_total = Fx_drive + Fx_drag + F_roll + Fx_slope;
    double Fy_total = F_lat    + Fy_slope;
    double Tz_total = Tz_drive;

    state_.ax    = Fx_total / m;
    state_.ay    = Fy_total / m;
    state_.alpha = Tz_total / params_.inertia;

    // ── STEP 9: Euler integration ─────────────────────────────────────────
    //
    //  Velocity:   v(t+dt) = v(t) + a(t)·dt
    //  Position (world frame):
    //    x += (vx·cos(yaw) − vy·sin(yaw)) · dt
    //    y += (vx·sin(yaw) + vy·cos(yaw)) · dt
    //    yaw += omega · dt
    //
    state_.vx    += state_.ax    * dt;
    state_.vy    += state_.ay    * dt;
    state_.omega += state_.alpha * dt;

    clamp_velocities();

    double cos_yaw = std::cos(state_.yaw);
    double sin_yaw = std::sin(state_.yaw);

    state_.x   += (state_.vx * cos_yaw - state_.vy * sin_yaw) * dt;
    state_.y   += (state_.vx * sin_yaw + state_.vy * cos_yaw) * dt;
    state_.yaw += state_.omega * dt;

    // Normalise yaw to (−π, π]
    while (state_.yaw >  M_PI) state_.yaw -= 2.0 * M_PI;
    while (state_.yaw < -M_PI) state_.yaw += 2.0 * M_PI;

    // ── STEP 10: Collision resolution ─────────────────────────────────────
    resolve_collisions();

    // ── STEP 11: Odometry dead-reckoning ──────────────────────────────────
    double v_odom  = 0.5 * (state_.omega_left + state_.omega_right) * R;
    double om_odom = (state_.omega_right - state_.omega_left) * R / L;
    odom_yaw_ += om_odom * dt;
    odom_x_   += v_odom * std::cos(odom_yaw_) * dt;
    odom_y_   += v_odom * std::sin(odom_yaw_) * dt;
}

// ─────────────────────────────────────────────────────────────────────────────
//  pid_step  —  run both PID controllers for one tick
// ─────────────────────────────────────────────────────────────────────────────
/*
 * FEATURE 1: PID CONTROLLER
 * ─────────────────────────
 *  pid_v_    tracks linear  speed:  error = v_target  − vx
 *  pid_omega_ tracks angular speed: error = omega_target − omega
 *
 *  The output of pid_v_ is a corrective forward force [N].
 *  The output of pid_omega_ is a corrective turning torque [N·m].
 *
 *  Anti-windup: if the integral grows too large (e.g. robot blocked by a wall),
 *  it is clamped to ±integral_limit to prevent overshoot when the block clears.
 *
 *  Reference: https://en.wikipedia.org/wiki/PID_controller
 */
void RobotDrive::pid_step(double dt, double & Fx_out, double & Tz_out)
{
    double v_err     = cmd_.v_target     - state_.vx;
    double omega_err = cmd_.omega_target - state_.omega;

    Fx_out = pid_v_.step(v_err,     dt);
    Tz_out = pid_omega_.step(omega_err, dt);
}

// ─────────────────────────────────────────────────────────────────────────────
//  sample_terrain  —  evaluate heightmap h(x,y) and its gradient
// ─────────────────────────────────────────────────────────────────────────────
/*
 * FEATURE 2: TERRAIN / SLOPE
 * ───────────────────────────
 * Heightmap (analytic, no lookup table needed):
 *
 *   h(x, y) = A · sin(kx · x) · sin(ky · y)
 *
 * Gradient (partial derivatives):
 *
 *   ∂h/∂x = A · kx · cos(kx · x) · sin(ky · y)   → state_.slope_x
 *   ∂h/∂y = A · ky · sin(kx · x) · cos(ky · y)   → state_.slope_y
 *
 * These are stored in state_ and used by apply_slope_gravity() and by
 * update_pose() which tilts the robot's TF transform to match the terrain.
 *
 * Reference: https://en.wikipedia.org/wiki/Rigid_body_dynamics
 */
void RobotDrive::sample_terrain()
{
    const double A  = params_.terrain_amplitude;
    const double kx = params_.terrain_kx;
    const double ky = params_.terrain_ky;
    const double x  = state_.x;
    const double y  = state_.y;

    state_.terrain_z = A * std::sin(kx * x) * std::sin(ky * y);
    state_.slope_x   = A * kx * std::cos(kx * x) * std::sin(ky * y);
    state_.slope_y   = A * ky * std::sin(kx * x) * std::cos(ky * y);
}

// ─────────────────────────────────────────────────────────────────────────────
//  apply_slope_gravity  —  decompose gravity along the slope surface
// ─────────────────────────────────────────────────────────────────────────────
/*
 * On a surface with gradient (∂h/∂x, ∂h/∂y) the slope angle is:
 *
 *   θ_x = atan(∂h/∂x)   (tilt in world-X direction)
 *   θ_y = atan(∂h/∂y)   (tilt in world-Y direction)
 *
 * The gravitational force component along the surface (downhill) is:
 *
 *   Fg_world_x = −m·g·sin(θ_x)  ≈  −m·g·(∂h/∂x)    (small angle approx)
 *   Fg_world_y = −m·g·sin(θ_y)  ≈  −m·g·(∂h/∂y)
 *
 * Project into the robot body frame using heading yaw:
 *
 *   Fx_body =  Fg_world_x·cos(yaw) + Fg_world_y·sin(yaw)
 *   Fy_body = −Fg_world_x·sin(yaw) + Fg_world_y·cos(yaw)
 */
void RobotDrive::apply_slope_gravity(double & Fx_body, double & Fy_body)
{
    const double g = 9.81;
    const double m = params_.mass;

    // World-frame gravity component along slope (small-angle: sin(θ) ≈ tan(θ))
    double Fg_wx = -m * g * state_.slope_x;
    double Fg_wy = -m * g * state_.slope_y;

    // Rotate world → body frame
    double cos_y = std::cos(state_.yaw);
    double sin_y = std::sin(state_.yaw);

    Fx_body =  Fg_wx * cos_y + Fg_wy * sin_y;
    Fy_body = -Fg_wx * sin_y + Fg_wy * cos_y;
}

// ─────────────────────────────────────────────────────────────────────────────
//  resolve_collisions  —  detect and respond to AABB obstacle overlaps
// ─────────────────────────────────────────────────────────────────────────────
/*
 * FEATURE 3: COLLISION DETECTION & RESPONSE
 * ──────────────────────────────────────────
 * For each BoxObstacle in the world:
 *
 *  Detection:
 *    The robot is modelled as a circle of radius params_.collision_radius.
 *    A circle-vs-AABB overlap test uses the nearest-point method:
 *      nearest point on box = clamp(robot_centre, box_min, box_max)
 *      overlap if dist(robot_centre, nearest_point) < collision_radius
 *
 *  Response:
 *    1. Positional correction: push the robot out along the contact normal
 *       by the penetration depth so it no longer overlaps.
 *
 *    2. Velocity reflection (normal component):
 *         v_n = dot(v_world, normal) * normal
 *         v_normal_new = −e · v_n        (e = restitution coefficient)
 *
 *    3. Tangential damping:
 *         v_t = v_world − v_n
 *         v_tangential_new = (1 − contact_friction) · v_t
 *
 *    The corrected world-frame velocity is then rotated back into the
 *    robot body frame.
 *
 * Reference: https://en.wikipedia.org/wiki/Rigid_body_dynamics  (Impulse section)
 */
void RobotDrive::resolve_collisions()
{
    const double r      = params_.collision_radius;
    const double e      = params_.restitution;
    const double mu_c   = params_.contact_friction;
    const double cos_y  = std::cos(state_.yaw);
    const double sin_y  = std::sin(state_.yaw);

    // Robot velocity in world frame
    double vwx = state_.vx * cos_y - state_.vy * sin_y;
    double vwy = state_.vx * sin_y + state_.vy * cos_y;

    bool hit = false;

    for (const auto & obs : obstacles_)
    {
        if (!obs.overlaps(state_.x, state_.y, r)) continue;

        hit = true;

        auto [nx, ny, depth] = obs.penetration(state_.x, state_.y, r);

        // 1. Positional correction — push out of overlap
        state_.x += nx * (depth + 1e-4);
        state_.y += ny * (depth + 1e-4);

        // 2. Decompose world velocity into normal and tangential components
        double v_n  = vwx * nx + vwy * ny;        // scalar projection onto normal

        // Only resolve if robot is moving INTO the surface (v_n < 0)
        if (v_n >= 0.0) continue;

        double vn_x = v_n * nx;                   // normal velocity vector
        double vn_y = v_n * ny;
        double vt_x = vwx - vn_x;                 // tangential velocity vector
        double vt_y = vwy - vn_y;

        // 3. Reflect normal component with restitution
        //    (e=0: dead stop, e=1: perfect bounce)
        double new_vn_x = -e * vn_x;
        double new_vn_y = -e * vn_y;

        // 4. Damp tangential component (contact friction)
        double new_vt_x = (1.0 - mu_c) * vt_x;
        double new_vt_y = (1.0 - mu_c) * vt_y;

        // Recombine and rotate back into body frame
        vwx = new_vn_x + new_vt_x;
        vwy = new_vn_y + new_vt_y;
    }

    if (hit)
    {
        // Rotate corrected world velocity back to body frame
        state_.vx =  vwx * cos_y + vwy * sin_y;
        state_.vy = -vwx * sin_y + vwy * cos_y;

        // Damp angular velocity on collision (prevents spinning into walls)
        state_.omega *= (1.0 - params_.contact_friction);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  clamp_velocities
// ─────────────────────────────────────────────────────────────────────────────
void RobotDrive::clamp_velocities()
{
    state_.vx    = std::clamp(state_.vx,    -params_.max_speed, params_.max_speed);
    state_.vy    = std::clamp(state_.vy,    -0.3,               0.3);
    state_.omega = std::clamp(state_.omega, -params_.max_omega,  params_.max_omega);
}

// ─────────────────────────────────────────────────────────────────────────────
//  publish_odometry
// ─────────────────────────────────────────────────────────────────────────────
void RobotDrive::publish_odometry()
{
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, odom_yaw_);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp    = get_clock()->now();
    odom.header.frame_id = "world";
    odom.child_frame_id  = "base_link";

    odom.pose.pose.position.x    = odom_x_;
    odom.pose.pose.position.y    = odom_y_;
    odom.pose.pose.position.z    = 0.0;
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x  = state_.vx;
    odom.twist.twist.angular.z = state_.omega;

    odom_pub_->publish(odom);
}

// ─────────────────────────────────────────────────────────────────────────────
//  publish_imu  —  simulated IMU with Gaussian noise
// ─────────────────────────────────────────────────────────────────────────────
void RobotDrive::publish_imu()
{
    constexpr double ACCEL_NOISE = 0.02;   // σ [m/s²]
    constexpr double GYRO_NOISE  = 0.005;  // σ [rad/s]

    sensor_msgs::msg::Imu imu;
    imu.header.stamp    = get_clock()->now();
    imu.header.frame_id = "base_link";

    // Include slope-induced gravity components in the accelerometer reading
    // (a real IMU would sense these as non-zero acceleration even when stationary)
    const double g = 9.81;
    imu.linear_acceleration.x = state_.ax
                                 + g * state_.slope_x   // tilt effect
                                 + gaussian_noise(ACCEL_NOISE);
    imu.linear_acceleration.y = state_.ay
                                 + g * state_.slope_y
                                 + gaussian_noise(ACCEL_NOISE);
    imu.linear_acceleration.z = g   // gravity in sensor Z when flat
                                 - g * (state_.slope_x * state_.slope_x
                                      + state_.slope_y * state_.slope_y) * 0.5;

    imu.angular_velocity.x = 0.0;
    imu.angular_velocity.y = 0.0;
    imu.angular_velocity.z = state_.omega + gaussian_noise(GYRO_NOISE);

    tf2::Quaternion q;
    q.setRPY(std::atan(state_.slope_y), -std::atan(state_.slope_x), state_.yaw);
    imu.orientation.x = q.x();
    imu.orientation.y = q.y();
    imu.orientation.z = q.z();
    imu.orientation.w = q.w();

    imu_pub_->publish(imu);
}
