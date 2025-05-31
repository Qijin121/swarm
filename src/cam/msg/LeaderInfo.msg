# Leader 信息消息定义
# 用于传递 Leader 机器人的位置和速度信息

# Leader 相对位置（在接收者坐标系下）
float64 rel_x      # Leader 相对接收者的 x 位置
float64 rel_y      # Leader 相对接收者的 y 位置

# Leader 速度（在 Leader 自身坐标系下）
float64 leader_vx  # Leader 的 x 方向速度
float64 leader_vy  # Leader 的 y 方向速度
