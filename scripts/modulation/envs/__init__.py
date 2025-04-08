from .tasks import RestrictedWsTask, RndStartRndGoalsTask, SimpleObstacleTask, DynamicObstacleTask, SplineTask
from .tasks_chained import PickNPlaceChainedTask, PickNPlaceDynChainedTask, DoorChainedTask, DrawerChainedTask, RoomDoorChainedTask, \
    BookstorePickNPlaceChainedTask
from .tasks_realworld import AisOfficeSplineTask, AisOfficeRoomDoorTask, AisOfficeDoorChainedTask, AisOfficePicknplace, AisOfficeDrawerChainedTask, PicknplaceRobotHall, DoorChainedTaskHall, DrawerChainedTaskHall, RoomDoorChainedTaskHall

_all_tasks = [RndStartRndGoalsTask, RestrictedWsTask, SimpleObstacleTask, SplineTask,
              PickNPlaceChainedTask, PickNPlaceDynChainedTask, DoorChainedTask, DrawerChainedTask, RoomDoorChainedTask,
              BookstorePickNPlaceChainedTask, DynamicObstacleTask,
              AisOfficeRoomDoorTask, AisOfficeSplineTask, AisOfficeDoorChainedTask, AisOfficePicknplace, AisOfficeDrawerChainedTask,
              PicknplaceRobotHall, DoorChainedTaskHall, DrawerChainedTaskHall, RoomDoorChainedTaskHall]
ALL_TASKS = {task.taskname().lower(): task for task in _all_tasks}
