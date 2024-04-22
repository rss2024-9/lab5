# HOW TF DO I RUN THIS
It is very simple...

## In Simulation
1. `ros2 launch localization localize.launch.xml`
2. `ros2 launch racecar_simulator simulate.launch.xml`
3. `ros2 launch wall_follower wall_follower.launch.xml` (or whatever gets your car movin)

## In Real Life
1. `teleop`
2. `ros2 launch localization localize_real_env.launch.xml`
3. `ros2 launch localization map.launch.xml`

For both scenarios, the order in which you run these commands is pretty critical. This is because the particle filter's sensor model relies on map data, so it subscribes to `/map`. Unfortunately ROS is poorly designed garbage so the topic is only notified once (at startup), and failing to launch the map server node _after_ the particle filter will cause the latter to miss it entirely. So, the sensor model would never receive map data and not work at all.

## In Real Life (Staff Solution)
1. `teleop`
2. `ros2 launch localization real_localize.launch.xml`
3. `ros2 launch localization localize_real_env.launch.xml`