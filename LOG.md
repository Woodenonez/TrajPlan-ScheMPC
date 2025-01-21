# Create (Jun. XX 2023) [TrajPlanFleet_ScheMPC_ScheFleetMPC]
To simulate a MPC-based collision-free fleet control with a high-level scheduler.

# 20231026 - (Savepoint src_20231026.zip)
The previous history is not recorded. This project is reinitialized today and should be uploaded to GitHub in two days (20231026). This project will be later used by the design project: *Time-Constrained Scheduling and Collision-Free Control for A Fleet of Mobile Robots*. (The savepoint is the version mixing the obstacle objects and inflation in the map, which is removed in the next version.)

- [x] ~~Reconstruction of the project and upload to GitHub.~~

# 20231031
The basic_map package is reconstructed to remove the dependency on the basic_obstacle package. Maps are exported to json files.
- [x] ~~test_map_and_graph~~

The basic_obstacle package is cleaned up (remove separate shapely geometry, merge into plain geometry).
MotionPlanInterface now has a current map and an inflated map.

The code is uploaded to GitHub.

# 20240112
The code is simplified by getting rid of interfaces.
- ~~[Abort] Merge the code with the design project.~~

# 20240125
Modify the linear sampling method in the local planner so that the reference time/speed can be None.

# 20240216
Add cost monitor and improve the interfaces of the controller and so on.
Return MPC parameters according to the cost monitor report.